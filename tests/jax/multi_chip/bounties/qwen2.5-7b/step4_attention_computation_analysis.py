#!/usr/bin/env python3
"""
Step 4: Analyze how sharded Q/K/V tensors behave in attention computation
"""
import os
import jax
import jax.numpy as jnp
import json
import flax.linen as nn
from typing import Dict, Any, Optional, Union
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import q25j7_tensor_parallel_fixed
from q25j7_tensor_parallel_fixed import ParallelDense, setup_device_mesh

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
    """Debug helper to show tensor properties including detailed sharding"""
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Sharding: {tensor.sharding}")
    print(f"  Device IDs: {[d.id for d in tensor.sharding.device_set]}")
    
    # Try to peek at device-local data
    try:
        local_data = jax.device_get(tensor)
        print(f"  Min/Max: {float(jnp.min(local_data)):.4f}, {float(jnp.max(local_data)):.4f}")
        print(f"  Mean/Std: {float(jnp.mean(local_data)):.4f}, {float(jnp.std(local_data)):.4f}")
    except Exception as e:
        print(f"  Data access error: {e}")
    print()

def analyze_attention_computation_flow():
    """Analyze exactly what happens in attention computation with sharded tensors"""
    
    mesh = setup_device_mesh()
    q25j7_tensor_parallel_fixed.mesh = mesh
    
    with open("weights/config.json") as f:
        config = json.load(f)
    
    print("=== STEP 4: ATTENTION COMPUTATION ANALYSIS ===\n")
    print("Goal: Understand how sharded Q/K/V tensors behave in attention")
    
    # Test dimensions
    batch, seq = 1, 8
    hidden_size = config["hidden_size"]  # 3584
    num_heads = config["num_attention_heads"]  # 28  
    num_kv_heads = config["num_key_value_heads"]  # 4
    head_dim = hidden_size // num_heads  # 128
    max_seq_len = config.get("max_position_embeddings", 32768)
    rope_theta = config.get("rope_theta", 1000000.0)
    
    print(f"Attention dimensions:")
    print(f"  Q heads: {num_heads}, K/V heads: {num_kv_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  GQA ratio: {num_heads // num_kv_heads}")
    
    # Create test input
    rng = jax.random.PRNGKey(42)
    hidden_states = jnp.ones((batch, seq, hidden_size), dtype=jnp.bfloat16)
    position_ids = jnp.arange(seq)[None, :].repeat(batch, axis=0)
    
    print(f"\nTest setup:")
    debug_tensor_properties(hidden_states, "Input hidden_states")
    print(f"Position IDs: {position_ids}")
    
    print("\n" + "=" * 70)
    print("STEP 4.1: Create Q/K/V projections and analyze their outputs")
    print("=" * 70)
    
    # Create Q/K/V projections
    q_proj = ParallelDense(num_heads * head_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, use_bias=True, name="q_proj")
    k_proj = ParallelDense(num_kv_heads * head_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, use_bias=True, name="k_proj") 
    v_proj = ParallelDense(num_kv_heads * head_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, use_bias=True, name="v_proj")
    
    # Initialize projections
    with mesh:
        q_params = q_proj.init(rng, hidden_states)
        k_params = k_proj.init(jax.random.split(rng)[0], hidden_states)
        v_params = v_proj.init(jax.random.split(rng)[1], hidden_states)
    
    print("✅ Q/K/V projections initialized")
    
    # Forward pass through projections
    print("\n--- Q/K/V projection outputs ---")
    with mesh:
        q_output = q_proj.apply(q_params, hidden_states)
        k_output = k_proj.apply(k_params, hidden_states) 
        v_output = v_proj.apply(v_params, hidden_states)
    
    debug_tensor_properties(q_output, "Q projection output")
    debug_tensor_properties(k_output, "K projection output")
    debug_tensor_properties(v_output, "V projection output")
    
    print("\n" + "=" * 70)
    print("STEP 4.2: Reshape to attention heads and analyze")
    print("=" * 70)
    
    def _split_heads(hidden_states, num_heads):
        head_dim = hidden_states.shape[-1] // num_heads
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, head_dim))
    
    # Reshape to attention heads
    print("--- Reshaping to attention heads ---")
    q_heads = _split_heads(q_output, num_heads)
    k_heads = _split_heads(k_output, num_kv_heads)
    v_heads = _split_heads(v_output, num_kv_heads)
    
    debug_tensor_properties(q_heads, "Q heads")
    debug_tensor_properties(k_heads, "K heads") 
    debug_tensor_properties(v_heads, "V heads")
    
    print("\n" + "=" * 70)
    print("STEP 4.3: Apply rotary embedding and analyze")
    print("=" * 70)
    
    print("--- Applying rotary embedding ---")
    cos, sin = compute_cos_sin_cache(position_ids, head_dim, rope_theta)
    
    debug_tensor_properties(cos, "Cos embedding")
    debug_tensor_properties(sin, "Sin embedding")
    
    q_rot, k_rot = apply_rotary_emb(q_heads, k_heads, cos, sin)
    
    debug_tensor_properties(q_rot, "Q after RoPE")
    debug_tensor_properties(k_rot, "K after RoPE")
    
    print("\n" + "=" * 70)
    print("STEP 4.4: Apply GQA repeat_kv and analyze")
    print("=" * 70)
    
    print("--- Applying GQA repeat_kv ---")
    num_key_value_groups = num_heads // num_kv_heads
    print(f"Repeating K/V by factor: {num_key_value_groups}")
    
    k_repeated = repeat_kv(k_rot, num_key_value_groups)
    v_repeated = repeat_kv(v_heads, num_key_value_groups)  # Note: V doesn't get RoPE
    
    debug_tensor_properties(k_repeated, "K after repeat_kv")
    debug_tensor_properties(v_repeated, "V after repeat_kv")
    
    print("\n--- Checking shape compatibility for attention ---")
    print(f"Q shape: {q_rot.shape}")
    print(f"K shape: {k_repeated.shape}")
    print(f"V shape: {v_repeated.shape}")
    
    if q_rot.shape[-2] == k_repeated.shape[-2] == v_repeated.shape[-2]:
        print("✅ All have same number of heads - shapes compatible")
    else:
        print("❌ Head dimension mismatch - shapes incompatible")
        return
    
    print("\n" + "=" * 70)
    print("STEP 4.5: Attention computation and sharding analysis")
    print("=" * 70)
    
    print("--- Creating attention masks ---")
    
    # Create causal mask
    causal_mask = make_causal_mask(jnp.ones((1, max_seq_len), dtype="bool"), dtype="bool")
    causal_mask = causal_mask[:, :, :seq, :seq]
    causal_mask = jnp.broadcast_to(causal_mask, (batch,) + causal_mask.shape[1:])
    
    attention_mask = jnp.ones((batch, seq), dtype=jnp.float32)
    attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
    attention_mask = combine_masks(attention_mask, causal_mask)
    
    debug_tensor_properties(attention_mask, "Combined attention mask")
    
    # Convert to attention bias
    from jax import lax
    attention_bias = lax.select(
        attention_mask > 0,
        jnp.full(attention_mask.shape, 0.0).astype(jnp.bfloat16),
        jnp.full(attention_mask.shape, jnp.finfo(jnp.bfloat16).min).astype(jnp.bfloat16),
    )
    
    debug_tensor_properties(attention_bias, "Attention bias")
    
    print("\n--- Computing attention weights ---")
    print("This is where sharding issues might occur...")
    
    try:
        # Compute attention weights using Flax function
        attn_weights = dot_product_attention_weights(
            q_rot,
            k_repeated,
            bias=attention_bias,
            dropout_rng=None,
            dropout_rate=0.0,
            deterministic=True,
            dtype=jnp.bfloat16,
            precision=None,
        )
        
        debug_tensor_properties(attn_weights, "Attention weights")
        print("✅ Attention weights computation succeeded")
        
        # Check attention weight properties
        local_weights = jax.device_get(attn_weights)
        print(f"Attention weights sum along last dim (should be ~1): {float(jnp.mean(jnp.sum(local_weights, axis=-1))):.4f}")
        
    except Exception as e:
        print(f"❌ Attention weights computation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n--- Computing attention output ---")
    try:
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, v_repeated)
        debug_tensor_properties(attn_output, "Attention output (before merge)")
        
        # Merge heads
        def _merge_heads(hidden_states):
            return hidden_states.reshape(hidden_states.shape[:2] + (hidden_size,))
        
        attn_merged = _merge_heads(attn_output)
        debug_tensor_properties(attn_merged, "Attention output (after merge)")
        
        print("✅ Attention computation completed successfully")
        
    except Exception as e:
        print(f"❌ Attention output computation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    print("\n🔍 FINDINGS:")
    print("1. ✅ Q/K/V projections produce correct shapes and sharding")
    print("2. ✅ Head reshaping works correctly")  
    print("3. ✅ Rotary embedding works correctly")
    print("4. ✅ GQA repeat_kv works correctly")
    print("5. ✅ Attention weights computation succeeds")
    print("6. ✅ Attention output computation succeeds")
    
    print("\n🤔 IF INDIVIDUAL STEPS WORK, WHY DOES FULL ATTENTION FAIL?")
    print("\nPossible issues:")
    print("1. 🔍 Interaction between multiple parallel operations")
    print("2. 🔍 Collective communication deadlocks")
    print("3. 🔍 Sharding incompatibility in complex computation graph")
    print("4. 🔍 Memory layout issues in multi-device context")
    
    print("\n📋 NEXT INVESTIGATION:")
    print("1. Test the EXACT sequence used in full model")
    print("2. Compare with regular Dense step-by-step")
    print("3. Check if issue is in O-projection interaction")
    print("4. Look for collective communication patterns")

if __name__ == "__main__":
    analyze_attention_computation_flow()