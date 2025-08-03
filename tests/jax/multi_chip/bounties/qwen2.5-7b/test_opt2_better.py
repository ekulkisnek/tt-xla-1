#!/usr/bin/env python3
"""
Better test script for JIT optimization
"""
import os
import sys
import time
import jax
import jax.numpy as jnp
import jax.random as random

# Disable x64 globally for faster inference
os.environ["JAX_ENABLE_X64"] = "0"

def test_jit_optimization():
    """Test that JIT compilation provides speedup."""
    print("Testing JIT optimization...")
    
    # Create a more realistic test function that mimics attention computation
    def attention_function(q, k, v, mask=None):
        # Simulate attention computation
        scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) / jnp.sqrt(q.shape[-1])
        if mask is not None:
            scores += mask
        probs = jax.nn.softmax(scores, axis=-1)
        output = jnp.einsum('bhqk,bhkd->bhqd', probs, v)
        return output
    
    # Create test data similar to attention computation
    batch_size = 1
    num_heads = 32
    seq_len = 512
    head_dim = 128
    
    key = random.PRNGKey(42)
    q = random.normal(key, shape=(batch_size, num_heads, seq_len, head_dim)).astype(jnp.bfloat16)
    k = random.normal(key, shape=(batch_size, num_heads, seq_len, head_dim)).astype(jnp.bfloat16)
    v = random.normal(key, shape=(batch_size, num_heads, seq_len, head_dim)).astype(jnp.bfloat16)
    mask = random.normal(key, shape=(batch_size, num_heads, seq_len, seq_len)).astype(jnp.float32) * -1e9
    
    print(f"Testing with batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}")
    
    # Test without JIT
    print("Testing without JIT...")
    start_time = time.perf_counter()
    result1 = attention_function(q, k, v, mask)
    no_jit_time = time.perf_counter() - start_time
    
    # Test with JIT
    print("Testing with JIT...")
    jit_func = jax.jit(attention_function)
    
    # Warm up
    _ = jit_func(q, k, v, mask)
    
    start_time = time.perf_counter()
    result2 = jit_func(q, k, v, mask)
    jit_time = time.perf_counter() - start_time
    
    # Compare results
    diff = float(jnp.max(jnp.abs(result1 - result2)))
    
    print(f"\nResults:")
    print(f"No JIT time: {no_jit_time:.6f} seconds")
    print(f"JIT time: {jit_time:.6f} seconds")
    print(f"Speedup: {no_jit_time / jit_time:.2f}x")
    print(f"Max difference: {diff:.6f}")
    
    if diff < 1e-3:
        print("✅ PASS: Results are consistent")
        return True, no_jit_time / jit_time
    else:
        print("❌ FAIL: Results differ")
        return False, 1.0

def test_multiple_forward_passes():
    """Test speedup with multiple forward passes (more realistic scenario)."""
    print("\n" + "="*50)
    print("Testing multiple forward passes...")
    
    # Create a function that simulates multiple layers (fixed number for JIT)
    def multi_layer_function(x, weights):
        result = x
        # Fixed number of layers for JIT compatibility
        for i in range(10):
            # Simulate linear layer + activation
            result = jnp.dot(result, weights[f"layer_{i}"])
            result = jax.nn.relu(result)
        return result
    
    # Create test data
    batch_size = 1
    hidden_size = 1024
    num_layers = 10
    
    key = random.PRNGKey(42)
    x = random.normal(key, shape=(batch_size, hidden_size)).astype(jnp.bfloat16)
    weights = {}
    for i in range(num_layers):
        weights[f"layer_{i}"] = random.normal(key, shape=(hidden_size, hidden_size)).astype(jnp.bfloat16)
    
    print(f"Testing with batch_size={batch_size}, hidden_size={hidden_size}, num_layers={num_layers}")
    print(f"Running 20 forward passes...")
    
    # Test without JIT
    print("Benchmarking without JIT...")
    times_no_jit = []
    for i in range(20):
        start_time = time.perf_counter()
        result = multi_layer_function(x, weights)
        times_no_jit.append(time.perf_counter() - start_time)
        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/20 runs without JIT")
    
    # Test with JIT
    print("Benchmarking with JIT...")
    jit_func = jax.jit(multi_layer_function)
    
    # Warm up
    _ = jit_func(x, weights)
    
    times_jit = []
    for i in range(20):
        start_time = time.perf_counter()
        result = jit_func(x, weights)
        times_jit.append(time.perf_counter() - start_time)
        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/20 runs with JIT")
    
    # Calculate statistics
    avg_time_no_jit = sum(times_no_jit) / len(times_no_jit)
    avg_time_jit = sum(times_jit) / len(times_jit)
    
    print(f"\nResults:")
    print(f"Average time without JIT: {avg_time_no_jit:.6f} seconds")
    print(f"Average time with JIT: {avg_time_jit:.6f} seconds")
    print(f"Speedup: {avg_time_no_jit / avg_time_jit:.2f}x")
    
    return avg_time_no_jit / avg_time_jit

if __name__ == "__main__":
    print("=== Testing OPTIMIZATION 2: JIT Compilation ===\n")
    
    # Test basic JIT speedup
    correctness_passed, speedup1 = test_jit_optimization()
    
    # Test multiple forward passes
    speedup2 = test_multiple_forward_passes()
    
    print(f"\n=== SUMMARY ===")
    print(f"Correctness: {'✅ PASS' if correctness_passed else '❌ FAIL'}")
    print(f"Single forward pass speedup: {speedup1:.2f}x")
    print(f"Multiple forward passes speedup: {speedup2:.2f}x")
    
    if correctness_passed and speedup1 > 1.0 and speedup2 > 1.0:
        print("🎉 OPTIMIZATION 2 SUCCESSFUL!")
        print(f"Average speedup: {((speedup1 + speedup2) / 2):.2f}x")
    else:
        print("⚠️  OPTIMIZATION 2 NEEDS ATTENTION") 