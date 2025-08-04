#!/usr/bin/env python3
"""
Debug Step 1: Check shapes in rotary embedding
"""
import os
import jax
import jax.numpy as jnp

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Debug version with prints."""
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
    print(f"  freqs shape: {freqs.shape}")
    
    t = jnp.arange(end, dtype=jnp.float32)
    print(f"  t shape: {t.shape}")
    
    freqs = jnp.outer(t, freqs).astype(jnp.float32)
    print(f"  freqs after outer: {freqs.shape}")
    
    freqs_cis = jnp.complex64(jnp.cos(freqs) + 1j * jnp.sin(freqs))
    print(f"  freqs_cis final: {freqs_cis.shape}")
    
    return freqs_cis

def debug_shapes():
    """Debug the shape issue in rotary embeddings"""
    
    print("=== DEBUGGING STEP 1 SHAPES ===\n")
    
    # Qwen config
    hidden_size = 3584
    num_heads = 28
    head_dim = hidden_size // num_heads  # 128
    max_seq_len = 32768
    rope_theta = 1000000.0
    
    print(f"Config:")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_heads: {num_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  max_seq_len: {max_seq_len}")
    print(f"  rope_theta: {rope_theta}")
    
    # Test tensor shapes
    batch, seq = 1, 8
    
    print(f"\nTest shapes:")
    print(f"  batch: {batch}, seq: {seq}")
    
    # Create test tensors
    hidden_states = jnp.ones((batch, seq, hidden_size), dtype=jnp.float32)
    position_ids = jnp.arange(seq)[None, :].repeat(batch, axis=0)
    
    print(f"  hidden_states: {hidden_states.shape}")
    print(f"  position_ids: {position_ids.shape}")
    
    # Simulate what happens in attention
    print(f"\n--- SIMULATING ATTENTION FLOW ---")
    
    # After projection and reshape (simulate wq output)
    q_proj_out = jnp.ones((batch, seq, hidden_size), dtype=jnp.float32)  # Full hidden size
    q_reshaped = q_proj_out.reshape(batch, seq, num_heads, head_dim)
    
    print(f"Q after projection: {q_proj_out.shape}")
    print(f"Q after reshape: {q_reshaped.shape}")
    
    # Precompute frequencies
    print(f"\n--- PRECOMPUTING FREQUENCIES ---")
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len * 2, theta=rope_theta)
    
    # Take frequencies for positions
    freqs_for_positions = jnp.take(freqs_cis, position_ids, axis=0)
    print(f"freqs_for_positions: {freqs_for_positions.shape}")
    
    # Check what the assertion expects
    print(f"\n--- CHECKING ASSERTION ---")
    print(f"q_reshaped.shape[1] (seq): {q_reshaped.shape[1]}")
    print(f"q_reshaped.shape[-1] (head_dim): {q_reshaped.shape[-1]}")
    print(f"Expected freqs shape: ({q_reshaped.shape[1]}, {q_reshaped.shape[-1]})")
    print(f"Actual freqs shape: {freqs_for_positions.shape}")
    
    if freqs_for_positions.shape == (q_reshaped.shape[1], q_reshaped.shape[-1]):
        print("✅ Shapes match!")
    else:
        print("❌ Shape mismatch!")
        print("Need to fix frequency computation or reshaping")

if __name__ == "__main__":
    debug_shapes()