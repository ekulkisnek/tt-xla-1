#!/usr/bin/env python3
"""
Step 6: Test exact attention sequence with REAL weights to see if that's the issue
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

# Import working functions
def compute_cos_sin_cache(position_ids, head_dim, rope_theta=1000000.0):
    pos = position_ids.astype(jnp.float32)
    dim = head_dim // 2
    freqs = 1.0 / (rope_theta ** (jnp.arange(0, dim, dtype=jnp.float32) / dim))
    t = pos[..., None] * freqs[None, None, :]
    cos = jnp.cos(t)
    sin = jnp.sin(t)
    cos = cos[..., None, :]
    sin = sin[..., None, :]
    return cos, sin

def apply_rotary_emb(q, k, cos, sin):
    half_dim = q.shape[-1] // 2
    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    k1, k2 = k[..., :half_dim], k[..., half_dim:]
    q_rot = jnp.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
    k_rot = jnp.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)
    return q_rot, k_rot

def repeat_kv(hidden_states: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :]
    hidden_states = jnp.repeat(hidden_states, n_rep, axis=3)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)

class MockLlamaConfig:
    def __init__(self, qwen_config):
        self.hidden_size = qwen_config["hidden_size"]
        self.num_attention_heads = qwen_config["num_attention_heads"] 
        self.num_key_value_heads = qwen_config.get("num_key_value_heads", self.num_attention_heads)
        self.max_sequence_length = qwen_config.get("max_position_embeddings", 32768)
        self.rope_theta = qwen_config.get("rope_theta", 10000.0)
        self.attn_pdrop = 0.0

class RealWeightsAttention(nn.Module):
    """Test the exact attention sequence with REAL weights"""
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
        
        # Full parallelization - what we want to achieve
        self.q_proj = ParallelDense(
            config.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
        )

        self.k_proj = ParallelDense(
            config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
        )

        self.v_proj = ParallelDense(
            config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
        )

        self.o_proj = ParallelDense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
        )

        self.causal_mask = make_causal_mask(
            jnp.ones((1, config.max_sequence_length), dtype="bool"), dtype="bool"
        )
        
        self.rope_theta = config.rope_theta

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(
            hidden_states.shape[:2] + (num_heads, self.head_dim)
        )

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        if attention_mask is None:
            attention_mask = jnp.ones((hidden_states.shape[0], hidden_states.shape[1]), dtype=jnp.float32)
        
        # Full Llama-style attention computation with real weights
        xq, xk, xv = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

        xq = self._split_heads(xq, self.num_heads)
        xk = self._split_heads(xk, self.num_key_value_heads)
        xv = self._split_heads(xv, self.num_key_value_heads)

        cos, sin = compute_cos_sin_cache(position_ids, self.head_dim, self.rope_theta)
        xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        query_length, key_length = xq.shape[1], xk.shape[1]

        causal_mask = self.causal_mask[:, :, :query_length, :key_length]
        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(
            causal_mask, (batch_size,) + causal_mask.shape[1:]
        )

        attention_mask = jnp.broadcast_to(
            jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape
        )
        attention_mask = combine_masks(attention_mask, causal_mask)

        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(
                self.dtype
            ),
        )

        xk = repeat_kv(xk, self.num_key_value_groups)
        xv = repeat_kv(xv, self.num_key_value_groups)

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
        attn_output = self.o_proj(attn_output)

        return attn_output, None

def test_with_real_weights():
    """Test the attention with real loaded weights"""
    
    mesh = setup_device_mesh()
    q25j7_tensor_parallel_fixed.mesh = mesh
    
    with open("weights/config.json") as f:
        config = json.load(f)
    
    print("=== STEP 6: REAL WEIGHTS TEST ===\n")
    print("Goal: Test if real weights cause the parallelization to fail")
    
    # Create attention with real structure
    llama_config = MockLlamaConfig(config)
    attention = RealWeightsAttention(config=llama_config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
    
    # Load tokenizer and create REAL input
    tokenizer = AutoTokenizer.from_pretrained("weights")
    test_input = "What is 2 + 2?"
    input_ids = tokenizer.encode(test_input, return_tensors="jax")
    
    print(f"Test input: '{test_input}'")
    print(f"Input IDs: {input_ids}")
    
    batch, seq = input_ids.shape
    hidden_size = config["hidden_size"]
    
    # Create realistic hidden states (simulate embedding output)
    # Use small random values similar to embedding outputs
    rng = jax.random.PRNGKey(42)
    hidden_states = jax.random.normal(rng, (batch, seq, hidden_size), dtype=jnp.bfloat16) * 0.1
    position_ids = jnp.arange(seq)[None, :].repeat(batch, axis=0)
    
    print(f"Simulated hidden states: {hidden_states.shape}")
    print(f"Hidden states stats: min={float(jnp.min(hidden_states)):.4f}, max={float(jnp.max(hidden_states)):.4f}")
    
    print("\n--- TESTING WITH RANDOM WEIGHTS ---")
    
    try:
        # Initialize with random weights
        params = attention.init(rng, hidden_states, position_ids=position_ids)
        print("✅ Random weight initialization successful")
        
        # Test forward pass with random weights
        with mesh:
            output_random, _ = attention.apply(params, hidden_states, position_ids=position_ids)
        
        print(f"✅ Random weights forward pass successful")
        print(f"Output shape: {output_random.shape}")
        print(f"Output stats: min={float(jnp.min(output_random)):.4f}, max={float(jnp.max(output_random)):.4f}")
        
    except Exception as e:
        print(f"❌ Random weights test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n--- ATTEMPTING TO LOAD REAL WEIGHTS ---")
    
    try:
        # Create a minimal model structure for weight loading
        class MinimalModel(nn.Module):
            config: Dict[str, Any]
            
            def setup(self):
                llama_config = MockLlamaConfig(self.config)
                self.attention = RealWeightsAttention(config=llama_config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name="layers_0/self_attn")
            
            def __call__(self, hidden_states, position_ids):
                output, _ = self.attention(hidden_states, position_ids=position_ids)
                return output
        
        model = MinimalModel(config=config)
        
        # Try to load just the first layer's attention weights
        print("Loading weights for first attention layer...")
        
        model_params = model.init(rng, hidden_states, position_ids=position_ids)
        print("✅ Model structure initialized")
        
        # Load the real weights
        full_params = load_params(model, "weights", jnp.bfloat16)
        print("✅ Real weights loaded successfully")
        
        # Debug: Check parameter structure
        print("Available parameter keys:")
        for key in full_params['params'].keys():
            print(f"  {key}")
            if 'self_attn' in key:
                print(f"    -> {key}")
        
        # Extract attention params - they are under layers_0 -> self_attn 
        if 'layers_0' in full_params['params']:
            layer_params = full_params['params']['layers_0']
            print(f"Layer 0 keys: {list(layer_params.keys())}")
            
            if 'self_attn' in layer_params:
                attn_params = layer_params['self_attn']
                print(f"Attention keys: {list(attn_params.keys())}")
                
                # We need to extract the right parameters and restructure them
                attention_params = {
                    'params': {
                        'q_proj': attn_params['q_proj'],
                        'k_proj': attn_params['k_proj'], 
                        'v_proj': attn_params['v_proj'],
                        'o_proj': attn_params['o_proj'],
                    }
                }
            else:
                print("❌ No self_attn in layers_0")
                return False
        else:
            print("❌ No layers_0 found")
            return False
        
        print("\n--- TESTING WITH REAL WEIGHTS ---")
        
        with mesh:
            output_real, _ = attention.apply(attention_params, hidden_states, position_ids=position_ids)
        
        print(f"✅ REAL WEIGHTS FORWARD PASS SUCCESSFUL!")
        print(f"Output shape: {output_real.shape}")
        print(f"Output stats: min={float(jnp.min(output_real)):.4f}, max={float(jnp.max(output_real)):.4f}")
        
        # Compare random vs real weights
        diff = jnp.abs(output_random - output_real)
        print(f"\nDifference between random and real weights:")
        print(f"  Max diff: {float(jnp.max(diff)):.4f}")
        print(f"  Mean diff: {float(jnp.mean(diff)):.4f}")
        
        if float(jnp.max(diff)) > 0.1:
            print("✅ Real weights produce significantly different output (expected)")
        else:
            print("⚠️ Real and random weights produce similar output (unexpected)")
        
        return True
        
    except Exception as e:
        print(f"❌ Real weights test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_embeddings():
    """Test if the issue is in embedding interactions"""
    
    print("\n" + "=" * 70)
    print("STEP 6B: EMBEDDING INTERACTION TEST")  
    print("=" * 70)
    
    mesh = setup_device_mesh()
    q25j7_tensor_parallel_fixed.mesh = mesh
    
    with open("weights/config.json") as f:
        config = json.load(f)
    
    print("\nGoal: Test if embedding -> attention flow reveals the issue")
    
    from q25j7_tensor_parallel_fixed import ParallelEmbed
    
    # Create minimal model with embedding + attention
    class EmbedAttentionModel(nn.Module):
        config: Dict[str, Any]
        
        def setup(self):
            c = self.config
            llama_config = MockLlamaConfig(c)
            self.embed_tokens = ParallelEmbed(c["vocab_size"], c["hidden_size"], dtype=jnp.bfloat16, name="embed_tokens")
            self.attention = RealWeightsAttention(config=llama_config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name="layers_0/self_attn")
        
        def __call__(self, input_ids):
            hidden_states = self.embed_tokens(input_ids)
            batch, seq = input_ids.shape
            position_ids = jnp.arange(seq)[None, :].repeat(batch, axis=0)
            attn_output, _ = self.attention(hidden_states, position_ids=position_ids)
            return attn_output
    
    model = EmbedAttentionModel(config=config)
    
    # Test with real input IDs
    tokenizer = AutoTokenizer.from_pretrained("weights")
    test_input = "What is 2 + 2?"
    input_ids = tokenizer.encode(test_input, return_tensors="jax")
    
    print(f"Test input: '{test_input}'")
    print(f"Input IDs: {input_ids}")
    
    try:
        rng = jax.random.PRNGKey(42)
        params = model.init(rng, input_ids)
        print("✅ Embed+Attention model initialized")
        
        # Load real weights
        full_params = load_params(model, "weights", jnp.bfloat16)
        print("✅ Real weights loaded")
        
        # Test forward pass
        with mesh:
            output = model.apply(full_params, input_ids)
        
        print(f"✅ EMBED+ATTENTION WITH REAL WEIGHTS SUCCESSFUL!")
        print(f"Output shape: {output.shape}")
        print(f"Output stats: min={float(jnp.min(output)):.4f}, max={float(jnp.max(output)):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Embed+Attention test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing attention with real weights to find the interaction issue...")
    
    # Test 1: Real weights in isolation
    result1 = test_with_real_weights()
    
    if result1:
        print("\n✅ ATTENTION WITH REAL WEIGHTS WORKS!")
        
        # Test 2: Embedding interaction
        result2 = test_with_embeddings()
        
        if result2:
            print("\n🤔 EMBEDDING+ATTENTION ALSO WORKS!")
            print("\nThis means the issue is even higher level - possibly:")
            print("1. Multiple decoder layers")
            print("2. LM head interaction") 
            print("3. Some aspect of the full model compilation")
            print("4. Different input data patterns")
            
        else:
            print("\n🎯 FOUND IT: Embedding+Attention interaction causes the issue!")
            
    else:
        print("\n🎯 FOUND IT: Real weights cause the attention to fail!")
        print("This suggests the issue is in weight loading or specific weight patterns.")