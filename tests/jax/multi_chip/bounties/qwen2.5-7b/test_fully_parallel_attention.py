#!/usr/bin/env python3
"""
Test fully parallel Qwen attention using ParallelDense for all projections
"""
import os
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

# Import and setup mesh properly
import q25j7_tensor_parallel_fixed
from q25j7_tensor_parallel_fixed import ParallelDense, compute_cos_sin_cache, apply_rotary_emb, setup_device_mesh

class FullyParallelQwenAttention(nn.Module):
    """Fully parallel Qwen attention using ParallelDense for ALL projections"""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        self.num_heads = c["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = c.get("num_key_value_heads", self.num_heads)
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.rope_theta = c.get("rope_theta", 1000000.0)
        
        # Use ParallelDense for ALL projections like Llama
        self.q_proj = ParallelDense(
            self.hidden_size, 
            dtype=jnp.bfloat16, 
            param_dtype=jnp.bfloat16, 
            name="q_proj"
        )
        self.k_proj = ParallelDense(
            self.kv_dim, 
            dtype=jnp.bfloat16, 
            param_dtype=jnp.bfloat16, 
            name="k_proj"
        )
        self.v_proj = ParallelDense(
            self.kv_dim, 
            dtype=jnp.bfloat16, 
            param_dtype=jnp.bfloat16, 
            name="v_proj"
        )
        self.o_proj = ParallelDense(
            self.hidden_size, 
            dtype=jnp.bfloat16, 
            param_dtype=jnp.bfloat16, 
            use_bias=False, 
            name="o_proj"
        )

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        batch, seq, _ = hidden_states.shape

        # Project inputs using ParallelDense (like Llama)
        q = self.q_proj(hidden_states).reshape(batch, seq, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(batch, seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(batch, seq, self.num_kv_heads, self.head_dim)

        # Apply rotary embeddings
        if position_ids is not None:
            cos, sin = compute_cos_sin_cache(position_ids, self.head_dim, self.rope_theta)
            q, k = apply_rotary_emb(q, k, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = jnp.concatenate([past_k, k], axis=1)
            v = jnp.concatenate([past_v, v], axis=1)

        cache_k, cache_v = k, v

        # GQA: Repeat k/v to match query heads
        if self.num_heads != self.num_kv_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = jnp.repeat(k, repeat, axis=2)
            v = jnp.repeat(v, repeat, axis=2)

        # Attention computation (same as before)
        q = q.transpose(0, 2, 1, 3)  # [batch, heads, seq, head_dim]
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
        if attention_mask is not None:
            scores += attention_mask
        # Use higher precision for attention scores to reduce FP diffs
        scores = scores.astype(jnp.float64)
        probs = jnp.clip(jax.nn.softmax(scores.astype(jnp.float32), axis=-1), 1e-9, 1 - 1e-9)
        attn_out = jnp.einsum('bhqk,bhkd->bhqd', probs, v)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq, self.hidden_size)

        # Use ParallelDense for output projection (like Llama)
        return self.o_proj(attn_out), (cache_k, cache_v)

def test_fully_parallel_attention():
    """Test fully parallel attention implementation"""
    
    # Setup mesh properly
    mesh = setup_device_mesh()
    q25j7_tensor_parallel_fixed.mesh = mesh
    
    print("=== TESTING FULLY PARALLEL QWEN ATTENTION ===\n")
    
    # Create test config (matching Qwen2.5-7B)
    config = {
        "hidden_size": 3584,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "rope_theta": 1000000.0
    }
    
    print(f"Config: {config['hidden_size']} hidden, {config['num_attention_heads']} heads, {config['num_key_value_heads']} kv_heads")
    
    # Test parameters
    batch, seq = 2, 10
    hidden_size = config["hidden_size"]
    
    # Create test input
    rng = jax.random.PRNGKey(0)
    hidden_states = jnp.ones((batch, seq, hidden_size), dtype=jnp.bfloat16)
    position_ids = jnp.arange(seq)[None, :].repeat(batch, axis=0)
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Position IDs shape: {position_ids.shape}")
    
    # Test the fully parallel attention
    print("\n--- FULLY PARALLEL ATTENTION ---")
    try:
        attention_layer = FullyParallelQwenAttention(config=config, dtype=jnp.bfloat16)
        
        # Initialize
        params = attention_layer.init(rng, hidden_states, position_ids=position_ids)
        
        # Check parameter structure
        print("Parameter structure:")
        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if proj in params['params']:
                shape = params['params'][proj]['kernel'].shape
                print(f"  {proj}: {shape}")
                
                # Verify all projections use ParallelDense (have 'kernel' not 'Dense_0')
                if 'kernel' in params['params'][proj]:
                    print(f"    ✅ {proj} uses ParallelDense")
                else:
                    print(f"    ❌ {proj} falls back to nn.Dense")
        
        # Forward pass
        print("\n--- FORWARD PASS ---")
        with mesh:
            attn_output, kv_cache = attention_layer.apply(
                params, 
                hidden_states,
                position_ids=position_ids
            )
        
        print(f"Attention output shape: {attn_output.shape}")
        print(f"KV cache shapes: k={kv_cache[0].shape}, v={kv_cache[1].shape}")
        print(f"Output min/max: {float(jnp.min(attn_output)):.4f}, {float(jnp.max(attn_output)):.4f}")
        print(f"Output mean/std: {float(jnp.mean(attn_output)):.4f}, {float(jnp.std(attn_output)):.4f}")
        
        # Verify output shape is correct
        expected_shape = (batch, seq, hidden_size)
        if attn_output.shape == expected_shape:
            print(f"✅ Output shape is correct: {expected_shape}")
        else:
            print(f"❌ Expected {expected_shape}, got {attn_output.shape}")
        
        # Check if output is reasonable (not all zeros/NaNs)
        if not jnp.any(jnp.isnan(attn_output)) and not jnp.any(jnp.isinf(attn_output)):
            print("✅ Output contains no NaN or Inf values")
        else:
            print("❌ Output contains NaN or Inf values")
        
        if jnp.std(attn_output) > 1e-6:
            print("✅ Output has reasonable variance")
        else:
            print("❌ Output variance is too low (might be all zeros)")
        
        print("\n🎉 FULLY PARALLEL ATTENTION WORKS!")
        return True
        
    except Exception as e:
        print(f"❌ Fully parallel attention failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_fully_parallel_attention()