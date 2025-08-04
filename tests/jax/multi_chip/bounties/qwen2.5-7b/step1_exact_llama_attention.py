#!/usr/bin/env python3
"""
Step 1: Test exact copy of Llama's attention with Qwen config
"""
import os
import jax
import jax.numpy as jnp
import json
import flax.linen as nn
from typing import Dict, Any, Optional, Union
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from jax import lax

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import q25j7_tensor_parallel_fixed
from q25j7_tensor_parallel_fixed import ParallelDense, setup_device_mesh, load_params
from transformers import AutoTokenizer

# Helper functions - use working Qwen rotary embedding, Llama repeat_kv
def compute_cos_sin_cache(position_ids, head_dim, rope_theta=1000000.0):
    """Qwen's working rotary embedding computation."""
    pos = position_ids.astype(jnp.float32)  # [batch, seq]
    dim = head_dim // 2
    freqs = 1.0 / (rope_theta ** (jnp.arange(0, dim, dtype=jnp.float32) / dim))
    t = pos[..., None] * freqs[None, None, :]
    cos = jnp.cos(t)
    sin = jnp.sin(t)
    # Expand for broadcasting: [batch, seq, 1, dim]
    cos = cos[..., None, :]
    sin = sin[..., None, :]
    return cos, sin

def apply_rotary_emb(q, k, cos, sin):
    """Qwen's working rotary embedding application."""
    # q, k: [batch, seq, heads, head_dim]
    # cos, sin: [batch, seq, 1, dim] where dim = head_dim // 2
    half_dim = q.shape[-1] // 2
    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    k1, k2 = k[..., :half_dim], k[..., half_dim:]
    # cos and sin are already [batch, seq, 1, dim], so they broadcast correctly
    q_rot = jnp.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
    k_rot = jnp.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)
    return q_rot, k_rot

def repeat_kv(hidden_states: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    """Repeat key/value states for GQA (exact Llama implementation)."""
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :]
    hidden_states = jnp.repeat(hidden_states, n_rep, axis=3)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)

# Create a mock config class like Llama's
class MockLlamaConfig:
    def __init__(self, qwen_config):
        # Map Qwen config to Llama-style config
        self.hidden_size = qwen_config["hidden_size"]
        self.num_attention_heads = qwen_config["num_attention_heads"] 
        self.num_key_value_heads = qwen_config.get("num_key_value_heads", self.num_attention_heads)
        self.max_sequence_length = qwen_config.get("max_position_embeddings", 32768)
        self.rope_theta = qwen_config.get("rope_theta", 10000.0)
        self.attn_pdrop = 0.0  # Llama default

class ExactLlamaAttention(nn.Module):
    """EXACT copy of Llama's FlaxLLaMAAttention with Qwen config mapping"""
    config: MockLlamaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # Use ParallelDense for all projections (exact Llama style)
        self.wq = ParallelDense(
            config.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        self.wk = ParallelDense(
            config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        self.wv = ParallelDense(
            config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        self.wo = ParallelDense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        self.causal_mask = make_causal_mask(
            jnp.ones((1, config.max_sequence_length), dtype="bool"), dtype="bool"
        )
        
        # Store config for rotary embedding computation
        self.rope_theta = config.rope_theta

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(
            hidden_states.shape[:2] + (num_heads, self.head_dim)
        )

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """EXACT copy of Llama's cache logic"""
        # detect if we're initializing by absence of existing cache data.
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable(
            "cache", "cached_key", jnp.zeros, key.shape, key.dtype
        )
        cached_value = self.variable(
            "cache", "cached_value", jnp.zeros, value.shape, value.dtype
        )
        cache_index = self.variable(
            "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)
        )

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        # EXACT copy of Llama's attention computation
        xq, xk, xv = (
            self.wq(hidden_states),
            self.wk(hidden_states),
            self.wv(hidden_states),
        )

        xq = self._split_heads(xq, self.num_heads)
        xk = self._split_heads(xk, self.num_key_value_heads)
        xv = self._split_heads(xv, self.num_key_value_heads)

        # Use Qwen-style rotary embedding computation
        cos, sin = compute_cos_sin_cache(position_ids, self.head_dim, self.rope_theta)
        xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        query_length, key_length = xq.shape[1], xk.shape[1]

        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                self.causal_mask,
                (0, 0, mask_shift, 0),
                (1, 1, query_length, max_decoder_length),
            )
        else:
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(
            causal_mask, (batch_size,) + causal_mask.shape[1:]
        )

        attention_mask = jnp.broadcast_to(
            jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape
        )
        attention_mask = combine_masks(attention_mask, causal_mask)

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.has_variable("cache", "cached_key") or init_cache:
            xk, xv, attention_mask = self._concatenate_to_cache(
                xk, xv, xq, attention_mask
            )

        # transform boolean mask into float mask
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(
                self.dtype
            ),
        )

        xk = repeat_kv(xk, self.num_key_value_groups)
        xv = repeat_kv(xv, self.num_key_value_groups)

        # usual dot product attention
        attn_weights = dot_product_attention_weights(
            xq,
            xk,
            bias=attention_bias,
            dropout_rng=None,
            dropout_rate=self.config.attn_pdrop,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=self.precision,
        )

        attn_output = jnp.einsum(
            "...hqk,...khd->...qhd", attn_weights, xv, precision=self.precision
        )
        attn_output = self._merge_heads(attn_output)
        attn_output = self.wo(attn_output)

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs

def test_exact_llama_attention():
    """Test Step 1: Exact Llama attention with Qwen dimensions"""
    
    # Setup
    mesh = setup_device_mesh()
    q25j7_tensor_parallel_fixed.mesh = mesh
    
    with open("weights/config.json") as f:
        qwen_config = json.load(f)
    
    print("=== STEP 1: TESTING EXACT LLAMA ATTENTION ===\n")
    print("Goal: Verify Llama's exact attention logic works with Qwen dimensions")
    
    # Create Llama-style config from Qwen config
    llama_config = MockLlamaConfig(qwen_config)
    
    print(f"Mapped config:")
    print(f"  hidden_size: {llama_config.hidden_size}")
    print(f"  num_attention_heads: {llama_config.num_attention_heads}")
    print(f"  num_key_value_heads: {llama_config.num_key_value_heads}")
    print(f"  rope_theta: {llama_config.rope_theta}")
    
    # Test input
    batch, seq = 1, 8
    hidden_size = llama_config.hidden_size
    
    rng = jax.random.PRNGKey(42)
    hidden_states = jnp.ones((batch, seq, hidden_size), dtype=jnp.bfloat16)
    attention_mask = jnp.ones((batch, seq), dtype=jnp.float32)
    position_ids = jnp.arange(seq)[None, :].repeat(batch, axis=0)
    
    print(f"\nTest input shapes:")
    print(f"  hidden_states: {hidden_states.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    print(f"  position_ids: {position_ids.shape}")
    
    # Test exact Llama attention
    print("\n--- TESTING EXACT LLAMA ATTENTION ---")
    try:
        attention_layer = ExactLlamaAttention(
            config=llama_config, 
            dtype=jnp.bfloat16, 
            param_dtype=jnp.bfloat16
        )
        
        # Initialize
        params = attention_layer.init(
            rng, 
            hidden_states, 
            attention_mask, 
            position_ids,
            deterministic=True
        )
        
        print("✅ Initialization successful")
        
        # Check parameter structure
        print("\nParameter structure:")
        for proj in ['wq', 'wk', 'wv', 'wo']:
            if proj in params['params']:
                shape = params['params'][proj]['kernel'].shape
                print(f"  {proj}: {shape}")
        
        # Forward pass
        print("\n--- FORWARD PASS ---")
        with mesh:
            outputs = attention_layer.apply(
                params, 
                hidden_states,
                attention_mask,
                position_ids,
                deterministic=True
            )
        
        attn_output = outputs[0]  # First element is attention output
        
        print(f"✅ Forward pass successful")
        print(f"Attention output shape: {attn_output.shape}")
        print(f"Expected shape: ({batch}, {seq}, {hidden_size})")
        
        if attn_output.shape == (batch, seq, hidden_size):
            print("✅ Output shape is correct")
        else:
            print("❌ Output shape is wrong")
            return False
        
        # Check output properties
        print(f"Output min/max: {float(jnp.min(attn_output)):.4f}, {float(jnp.max(attn_output)):.4f}")
        print(f"Output mean/std: {float(jnp.mean(attn_output)):.4f}, {float(jnp.std(attn_output)):.4f}")
        
        # Check for NaN/Inf
        if jnp.any(jnp.isnan(attn_output)) or jnp.any(jnp.isinf(attn_output)):
            print("❌ Output contains NaN or Inf")
            return False
        else:
            print("✅ Output is numerically stable")
        
        # Check reasonable variance
        if jnp.std(attn_output) > 1e-6:
            print("✅ Output has reasonable variance")
        else:
            print("❌ Output variance too low (might be all zeros)")
            return False
        
        print("\n🎉 STEP 1 SUCCESS: Exact Llama attention works with Qwen dimensions!")
        return True
        
    except Exception as e:
        print(f"❌ Step 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_exact_llama_attention()
    
    if success:
        print("\n✅ STEP 1 COMPLETE: Ready for Step 2 (minimal Qwen adaptations)")
    else:
        print("\n❌ STEP 1 FAILED: Need to debug exact Llama implementation")