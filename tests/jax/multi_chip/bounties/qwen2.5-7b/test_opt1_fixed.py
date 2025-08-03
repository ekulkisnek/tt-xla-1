#!/usr/bin/env python3
"""
Fixed test script to compare original vs optimized rotary embeddings
"""
import os
import sys
import time
import jax
import jax.numpy as jnp
import jax.random as random

# Disable x64 globally for faster inference
os.environ["JAX_ENABLE_X64"] = "0"

def test_rotary_embeddings_functions():
    """Test that both rotary embedding functions produce the same results."""
    print("Testing rotary embeddings optimization...")
    
    # Original function (from q25j7fast1)
    def compute_cos_sin_cache_original(position_ids, head_dim, rope_theta=1000000.0):
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

    def apply_rotary_emb_original(q, k, cos, sin):
        # q, k: [batch, seq, heads, head_dim]
        # cos, sin: [batch, seq, 1, dim] where dim = head_dim // 2
        half_dim = q.shape[-1] // 2
        q1, q2 = q[..., :half_dim], q[..., half_dim:]
        k1, k2 = k[..., :half_dim], k[..., half_dim:]
        # cos and sin are already [batch, seq, 1, dim], so they broadcast correctly
        q_rot = jnp.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
        k_rot = jnp.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)
        return q_rot, k_rot

    # Optimized function - using the same approach but precomputed
    def precompute_freqs_cis_optimized(dim: int, end: int, theta: float = 500000.0):
        """Precompute rotary embedding frequencies."""
        # Create frequency bands: [dim//2]
        freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)] / dim))
        
        # Create position indices: [end]
        t = jnp.arange(end)
        
        # Outer product: [end, dim//2]
        freqs = jnp.outer(t, freqs)
        
        # Compute cos and sin: [end, dim//2]
        cos = jnp.cos(freqs)
        sin = jnp.sin(freqs)
        
        return cos, sin

    def apply_rotary_emb_optimized(xq: jnp.ndarray, xk: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray, dtype: jnp.dtype = jnp.float32):
        """Apply rotary embeddings using precomputed cos/sin."""
        # cos, sin: [seq_len, dim//2]
        # xq, xk: [batch, seq, heads, head_dim]
        
        # Expand cos/sin for broadcasting: [seq_len, 1, dim//2]
        cos = cos[..., None, :]  # [seq_len, 1, dim//2]
        sin = sin[..., None, :]  # [seq_len, 1, dim//2]
        
        # Apply the same logic as original
        half_dim = xq.shape[-1] // 2
        q1, q2 = xq[..., :half_dim], xq[..., half_dim:]
        k1, k2 = xk[..., :half_dim], xk[..., half_dim:]
        
        # cos and sin will broadcast to [batch, seq, heads, dim//2]
        q_rot = jnp.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
        k_rot = jnp.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)
        
        return q_rot.astype(dtype), k_rot.astype(dtype)

    # Test parameters
    batch_size = 1
    seq_len = 128
    num_heads = 8
    head_dim = 64
    rope_theta = 1000000.0
    
    # Create test data
    key = random.PRNGKey(42)
    position_ids = jnp.array([[i for i in range(seq_len)]])
    q = random.normal(key, shape=(batch_size, seq_len, num_heads, head_dim)).astype(jnp.bfloat16)
    k = random.normal(key, shape=(batch_size, seq_len, num_heads, head_dim)).astype(jnp.bfloat16)
    
    print(f"Testing with batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}")
    
    # Test original method
    print("Testing original method...")
    start_time = time.perf_counter()
    cos, sin = compute_cos_sin_cache_original(position_ids, head_dim, rope_theta)
    q_orig, k_orig = apply_rotary_emb_original(q, k, cos, sin)
    original_time = time.perf_counter() - start_time
    
    # Test optimized method
    print("Testing optimized method...")
    start_time = time.perf_counter()
    # Precompute frequencies (this would be done once at model initialization)
    cos_pre, sin_pre = precompute_freqs_cis_optimized(head_dim, seq_len * 2, rope_theta)
    # Apply rotary embeddings (this would be done during inference)
    cos_lookup = jnp.take(cos_pre, position_ids[0], axis=0)  # [seq_len, dim//2]
    sin_lookup = jnp.take(sin_pre, position_ids[0], axis=0)  # [seq_len, dim//2]
    q_opt, k_opt = apply_rotary_emb_optimized(q, k, cos_lookup, sin_lookup, jnp.bfloat16)
    optimized_time = time.perf_counter() - start_time
    
    # Compare outputs
    q_diff = jnp.abs(q_orig - q_opt)
    k_diff = jnp.abs(k_orig - k_opt)
    max_q_diff = jnp.max(q_diff)
    max_k_diff = jnp.max(k_diff)
    mean_q_diff = jnp.mean(q_diff)
    mean_k_diff = jnp.mean(k_diff)
    
    print(f"\nResults:")
    print(f"Original time: {original_time:.4f} seconds")
    print(f"Optimized time: {optimized_time:.4f} seconds")
    print(f"Speedup: {original_time / optimized_time:.2f}x")
    print(f"Max Q difference: {max_q_diff:.6f}")
    print(f"Max K difference: {max_k_diff:.6f}")
    print(f"Mean Q difference: {mean_q_diff:.6f}")
    print(f"Mean K difference: {mean_k_diff:.6f}")
    
    # Check if differences are acceptable
    # Use relative error instead of absolute error
    q_relative_error = jnp.max(jnp.abs(q_diff) / (jnp.abs(q_orig) + 1e-8))
    k_relative_error = jnp.max(jnp.abs(k_diff) / (jnp.abs(k_orig) + 1e-8))
    
    print(f"Max Q relative error: {q_relative_error:.6f}")
    print(f"Max K relative error: {k_relative_error:.6f}")
    
    if q_relative_error < 1e-2 and k_relative_error < 1e-2:  # 1% relative error tolerance
        print("✅ PASS: Outputs are sufficiently close")
        return True, original_time / optimized_time
    else:
        print("❌ FAIL: Outputs differ too much")
        return False, 1.0

def test_speed_with_multiple_runs():
    """Test speed improvement with multiple runs."""
    print("\n" + "="*50)
    print("Testing speed improvement with multiple runs...")
    
    # Original function (from q25j7fast1)
    def compute_cos_sin_cache_original(position_ids, head_dim, rope_theta=1000000.0):
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

    def apply_rotary_emb_original(q, k, cos, sin):
        # q, k: [batch, seq, heads, head_dim]
        # cos, sin: [batch, seq, 1, dim] where dim = head_dim // 2
        half_dim = q.shape[-1] // 2
        q1, q2 = q[..., :half_dim], q[..., half_dim:]
        k1, k2 = k[..., :half_dim], k[..., half_dim:]
        # cos and sin are already [batch, seq, 1, dim], so they broadcast correctly
        q_rot = jnp.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
        k_rot = jnp.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)
        return q_rot, k_rot

    # Optimized function - using the same approach but precomputed
    def precompute_freqs_cis_optimized(dim: int, end: int, theta: float = 500000.0):
        """Precompute rotary embedding frequencies."""
        # Create frequency bands: [dim//2]
        freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)] / dim))
        
        # Create position indices: [end]
        t = jnp.arange(end)
        
        # Outer product: [end, dim//2]
        freqs = jnp.outer(t, freqs)
        
        # Compute cos and sin: [end, dim//2]
        cos = jnp.cos(freqs)
        sin = jnp.sin(freqs)
        
        return cos, sin

    def apply_rotary_emb_optimized(xq: jnp.ndarray, xk: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray, dtype: jnp.dtype = jnp.float32):
        """Apply rotary embeddings using precomputed cos/sin."""
        # cos, sin: [seq_len, dim//2]
        # xq, xk: [batch, seq, heads, head_dim]
        
        # Expand cos/sin for broadcasting: [seq_len, 1, dim//2]
        cos = cos[..., None, :]  # [seq_len, 1, dim//2]
        sin = sin[..., None, :]  # [seq_len, 1, dim//2]
        
        # Apply the same logic as original
        half_dim = xq.shape[-1] // 2
        q1, q2 = xq[..., :half_dim], xq[..., half_dim:]
        k1, k2 = xk[..., :half_dim], xk[..., half_dim:]
        
        # cos and sin will broadcast to [batch, seq, heads, dim//2]
        q_rot = jnp.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
        k_rot = jnp.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)
        
        return q_rot.astype(dtype), k_rot.astype(dtype)
    
    # Test parameters
    batch_size = 1
    seq_len = 512
    num_heads = 16
    head_dim = 128
    rope_theta = 1000000.0
    num_runs = 10
    
    # Create test data
    key = random.PRNGKey(42)
    position_ids = jnp.array([[i for i in range(seq_len)]])
    q = random.normal(key, shape=(batch_size, seq_len, num_heads, head_dim)).astype(jnp.bfloat16)
    k = random.normal(key, shape=(batch_size, seq_len, num_heads, head_dim)).astype(jnp.bfloat16)
    
    print(f"Testing with batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}")
    print(f"Running {num_runs} iterations...")
    
    # Original method timing
    print("Benchmarking original method...")
    original_times = []
    for i in range(num_runs):
        start_time = time.perf_counter()
        cos, sin = compute_cos_sin_cache_original(position_ids, head_dim, rope_theta)
        q_orig, k_orig = apply_rotary_emb_original(q, k, cos, sin)
        original_times.append(time.perf_counter() - start_time)
        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/{num_runs} original runs")
    
    # Optimized method timing
    print("Benchmarking optimized method...")
    optimized_times = []
    # Precompute frequencies once
    cos_pre, sin_pre = precompute_freqs_cis_optimized(head_dim, seq_len * 2, rope_theta)
    for i in range(num_runs):
        start_time = time.perf_counter()
        cos_lookup = jnp.take(cos_pre, position_ids[0], axis=0)  # [seq_len, dim//2]
        sin_lookup = jnp.take(sin_pre, position_ids[0], axis=0)  # [seq_len, dim//2]
        q_opt, k_opt = apply_rotary_emb_optimized(q, k, cos_lookup, sin_lookup, jnp.bfloat16)
        optimized_times.append(time.perf_counter() - start_time)
        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/{num_runs} optimized runs")
    
    # Calculate statistics
    original_avg = sum(original_times) / len(original_times)
    optimized_avg = sum(optimized_times) / len(optimized_times)
    original_std = (sum((t - original_avg) ** 2 for t in original_times) / len(original_times)) ** 0.5
    optimized_std = (sum((t - optimized_avg) ** 2 for t in optimized_times) / len(optimized_times)) ** 0.5
    
    print(f"\nResults:")
    print(f"Original: {original_avg:.6f} ± {original_std:.6f} seconds")
    print(f"Optimized: {optimized_avg:.6f} ± {optimized_std:.6f} seconds")
    print(f"Speedup: {original_avg / optimized_avg:.2f}x")
    
    return original_avg / optimized_avg

if __name__ == "__main__":
    print("=== Testing OPTIMIZATION 1: Precomputed Rotary Embeddings ===\n")
    
    # Test correctness
    correctness_passed, speedup1 = test_rotary_embeddings_functions()
    
    # Test speed improvement
    speedup2 = test_speed_with_multiple_runs()
    
    print(f"\n=== SUMMARY ===")
    print(f"Correctness: {'✅ PASS' if correctness_passed else '❌ FAIL'}")
    print(f"Speedup (single run): {speedup1:.2f}x")
    print(f"Speedup (multiple runs): {speedup2:.2f}x")
    
    if correctness_passed and speedup1 > 1.0 and speedup2 > 1.0:
        print("🎉 OPTIMIZATION 1 SUCCESSFUL!")
        print(f"Average speedup: {((speedup1 + speedup2) / 2):.2f}x")
    else:
        print("⚠️  OPTIMIZATION 1 NEEDS ATTENTION") 