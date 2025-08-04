#!/usr/bin/env python3
"""
Step 5: Test the EXACT sequence used in the full model to find the interaction issue
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

# Import working functions from previous steps
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

def debug_tensor_properties(tensor, name):
    """Debug helper"""
    print(f"{name}: shape={tensor.shape}, sharding={tensor.sharding}")

class MockLlamaConfig:
    def __init__(self, qwen_config):
        self.hidden_size = qwen_config["hidden_size"]
        self.num_attention_heads = qwen_config["num_attention_heads"] 
        self.num_key_value_heads = qwen_config.get("num_key_value_heads", self.num_attention_heads)
        self.max_sequence_length = qwen_config.get("max_position_embeddings", 32768)
        self.rope_theta = qwen_config.get("rope_theta", 10000.0)
        self.attn_pdrop = 0.0

class ExactSequenceAttention(nn.Module):
    """Test the EXACT sequence that happens in the full model"""
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
        
        # Q/K/V use ParallelDense (this is where the issue lies)
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

        # O projection ALSO uses ParallelDense
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
        print("\n=== EXACT SEQUENCE ATTENTION COMPUTATION ===")
        
        if attention_mask is None:
            attention_mask = jnp.ones((hidden_states.shape[0], hidden_states.shape[1]), dtype=jnp.float32)
        
        debug_tensor_properties(hidden_states, "Input hidden_states")
        
        print("\n--- Step 1: Q/K/V Projections (ParallelDense) ---")
        xq = self.q_proj(hidden_states)
        debug_tensor_properties(xq, "Q projection output")
        
        xk = self.k_proj(hidden_states)
        debug_tensor_properties(xk, "K projection output")
        
        xv = self.v_proj(hidden_states)
        debug_tensor_properties(xv, "V projection output")

        print("\n--- Step 2: Head Reshaping ---")
        xq = self._split_heads(xq, self.num_heads)
        xk = self._split_heads(xk, self.num_key_value_heads)
        xv = self._split_heads(xv, self.num_key_value_heads)
        
        debug_tensor_properties(xq, "Q after split_heads")
        debug_tensor_properties(xk, "K after split_heads")
        debug_tensor_properties(xv, "V after split_heads")

        print("\n--- Step 3: Rotary Embedding ---")
        cos, sin = compute_cos_sin_cache(position_ids, self.head_dim, self.rope_theta)
        xq, xk = apply_rotary_emb(xq, xk, cos, sin)
        
        debug_tensor_properties(xq, "Q after RoPE")
        debug_tensor_properties(xk, "K after RoPE")

        query_length, key_length = xq.shape[1], xk.shape[1]

        print("\n--- Step 4: Causal Mask ---")
        causal_mask = self.causal_mask[:, :, :query_length, :key_length]
        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(
            causal_mask, (batch_size,) + causal_mask.shape[1:]
        )

        attention_mask = jnp.broadcast_to(
            jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape
        )
        attention_mask = combine_masks(attention_mask, causal_mask)

        print("\n--- Step 5: Attention Bias ---")
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(
                self.dtype
            ),
        )
        debug_tensor_properties(attention_bias, "Attention bias")

        print("\n--- Step 6: GQA Repeat K/V ---")
        xk = repeat_kv(xk, self.num_key_value_groups)
        xv = repeat_kv(xv, self.num_key_value_groups)
        
        debug_tensor_properties(xk, "K after repeat_kv")
        debug_tensor_properties(xv, "V after repeat_kv")

        print("\n--- Step 7: Attention Weights ---")
        print("⚠️ This is where collective communication might cause issues...")
        
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
        
        debug_tensor_properties(attn_weights, "Attention weights")

        print("\n--- Step 8: Attention Output ---")
        attn_output = jnp.einsum(
            "...hqk,...khd->...qhd", attn_weights, xv, precision=self.precision
        )
        debug_tensor_properties(attn_output, "Attention output (before merge)")
        
        attn_output = self._merge_heads(attn_output)
        debug_tensor_properties(attn_output, "Attention output (after merge)")
        
        print("\n--- Step 9: O Projection (ParallelDense) ---")
        print("⚠️ This is where Q/K/V sharding meets O projection sharding...")
        
        attn_output = self.o_proj(attn_output)
        debug_tensor_properties(attn_output, "Final attention output")

        print("\n=== ATTENTION SEQUENCE COMPLETE ===")
        return attn_output, None

def test_exact_model_sequence():
    """Test the exact sequence to isolate the interaction issue"""
    
    mesh = setup_device_mesh()
    q25j7_tensor_parallel_fixed.mesh = mesh
    
    with open("weights/config.json") as f:
        config = json.load(f)
    
    print("=== STEP 5: EXACT MODEL SEQUENCE TEST ===\n")
    print("Goal: Test the exact attention sequence to find interaction issues")
    
    # Create the exact attention used in the failing model
    llama_config = MockLlamaConfig(config)
    attention = ExactSequenceAttention(config=llama_config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
    
    # Test input (same as always)
    batch, seq = 1, 8
    hidden_size = config["hidden_size"]
    
    hidden_states = jnp.ones((batch, seq, hidden_size), dtype=jnp.bfloat16)
    position_ids = jnp.arange(seq)[None, :].repeat(batch, axis=0)
    
    print(f"Test input:")
    print(f"  Hidden states: {hidden_states.shape}")
    print(f"  Position IDs: {position_ids.shape}")
    
    print("\n--- TESTING ATTENTION SEQUENCE ---")
    
    try:
        # Initialize
        rng = jax.random.PRNGKey(42)
        params = attention.init(rng, hidden_states, position_ids=position_ids)
        print("✅ Attention initialization successful")
        
        # Forward pass
        print("\n--- FORWARD PASS ---")
        with mesh:
            output, _ = attention.apply(params, hidden_states, position_ids=position_ids)
        
        print(f"\n✅ FORWARD PASS SUCCESSFUL!")
        print(f"Final output shape: {output.shape}")
        print(f"Final output stats: min={float(jnp.min(output)):.4f}, max={float(jnp.max(output)):.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FORWARD PASS FAILED: {e}")
        print("\nThis is where the issue occurs in the full model!")
        
        import traceback
        traceback.print_exc()
        
        # Try to determine at which step it failed
        print("\n🔍 DEBUGGING: Let's narrow down which step failed...")
        
        return False

def test_step_by_step_comparison():
    """Compare step-by-step with working O-only version"""
    
    print("\n" + "=" * 70)
    print("STEP 5B: STEP-BY-STEP COMPARISON")
    print("=" * 70)
    
    print("\nGoal: Compare the failing full-parallel vs working O-only step by step")
    
    # TODO: If the exact sequence test fails, we can implement a detailed comparison here
    # This would help identify exactly at which point the computation diverges
    
    print("(This test will be implemented if the exact sequence test reveals the failure point)")

if __name__ == "__main__":
    print("Testing the exact attention sequence used in the failing model...")
    
    result = test_exact_model_sequence()
    
    if result:
        print("\n🤔 SURPRISING: The exact sequence works in isolation!")
        print("This suggests the issue is in model-level interactions, not attention-level.")
        print("\n📋 NEXT STEPS:")
        print("1. Test with multiple layers")
        print("2. Test with embedding/LM head interactions")
        print("3. Test with exact weight loading")
    else:
        print("\n🎯 FOUND IT: The exact sequence fails!")
        print("Now we know exactly where to investigate.")
        
        # Could add step-by-step comparison here
        test_step_by_step_comparison()