#!/usr/bin/env python3
"""
Test script to compare optimization 1 vs optimization 2 (JIT with donate_argnames)
"""
import os
import sys
import time
import jax
import jax.numpy as jnp
import jax.random as random
import gc

# Disable x64 globally for faster inference
os.environ["JAX_ENABLE_X64"] = "0"

def test_jit_donate_argnames():
    """Test that JIT with donate_argnames works correctly and provides speedup."""
    print("Testing JIT with donate_argnames optimization...")
    
    # Create a simple test function that mimics model.apply
    def test_function(params, input_data, return_dict=True):
        # Simulate some computation
        result = jnp.sum(params["weights"] * input_data) + params["bias"]
        if return_dict:
            return {"output": result, "metadata": jnp.array(1.0)}  # Use numeric instead of string
        return result
    
    # Create test data
    key = random.PRNGKey(42)
    params = {
        "weights": random.normal(key, shape=(1000, 1000)).astype(jnp.bfloat16),
        "bias": jnp.array(1.0, dtype=jnp.bfloat16)
    }
    input_data = random.normal(key, shape=(1000,)).astype(jnp.bfloat16)
    
    print(f"Testing with params shape: {params['weights'].shape}")
    
    # Test without JIT
    print("Testing without JIT...")
    start_time = time.perf_counter()
    result1 = test_function(params, input_data)
    no_jit_time = time.perf_counter() - start_time
    
    # Test with JIT (no donate)
    print("Testing with JIT (no donate)...")
    jit_func = jax.jit(test_function, static_argnames=['return_dict'])
    start_time = time.perf_counter()
    result2 = jit_func(params, input_data)
    jit_time = time.perf_counter() - start_time
    
    # Test with JIT + donate_argnames
    print("Testing with JIT + donate_argnames...")
    jit_donate_func = jax.jit(
        test_function, 
        static_argnames=['return_dict'],
        donate_argnames=['params']
    )
    start_time = time.perf_counter()
    result3 = jit_donate_func(params, input_data)
    jit_donate_time = time.perf_counter() - start_time
    
    # Compare results
    print(f"\nResults:")
    print(f"No JIT time: {no_jit_time:.6f} seconds")
    print(f"JIT time: {jit_time:.6f} seconds")
    print(f"JIT + donate time: {jit_donate_time:.6f} seconds")
    print(f"JIT speedup: {no_jit_time / jit_time:.2f}x")
    print(f"JIT + donate speedup: {no_jit_time / jit_donate_time:.2f}x")
    print(f"Donate additional speedup: {jit_time / jit_donate_time:.2f}x")
    
    # Check if results are the same
    diff1 = float(jnp.abs(result1["output"] - result2["output"]))
    diff2 = float(jnp.abs(result1["output"] - result3["output"]))
    
    print(f"Max difference (no JIT vs JIT): {diff1:.6f}")
    print(f"Max difference (no JIT vs JIT + donate): {diff2:.6f}")
    
    if diff1 < 1e-3 and diff2 < 1e-3:
        print("✅ PASS: All results are consistent")
        return True, no_jit_time / jit_donate_time
    else:
        print("❌ FAIL: Results differ")
        return False, 1.0

def test_memory_efficiency():
    """Test memory efficiency improvement with donate_argnames."""
    print("\n" + "="*50)
    print("Testing memory efficiency improvement...")
    
    import psutil
    
    def memory_intensive_function(params, input_data, iterations=100):
        # Simulate memory-intensive computation
        result = input_data
        for i in range(iterations):
            result = jnp.dot(params["weights"], result) + params["bias"]
        return result
    
    # Create larger test data
    key = random.PRNGKey(42)
    params = {
        "weights": random.normal(key, shape=(2000, 2000)).astype(jnp.bfloat16),
        "bias": random.normal(key, shape=(2000,)).astype(jnp.bfloat16)
    }
    input_data = random.normal(key, shape=(2000,)).astype(jnp.bfloat16)
    
    print(f"Testing with params shape: {params['weights'].shape}")
    print(f"Running 10 iterations...")
    
    # Test without donate_argnames
    print("Benchmarking without donate_argnames...")
    jit_func = jax.jit(memory_intensive_function)
    
    initial_mem = psutil.virtual_memory().used / (1024**3)
    times_no_donate = []
    
    for i in range(10):
        start_time = time.perf_counter()
        result = jit_func(params, input_data)
        times_no_donate.append(time.perf_counter() - start_time)
        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/10 runs without donate")
    
    peak_mem_no_donate = psutil.virtual_memory().used / (1024**3)
    
    # Test with donate_argnames
    print("Benchmarking with donate_argnames...")
    jit_donate_func = jax.jit(
        memory_intensive_function,
        donate_argnames=['params']
    )
    
    # Reset memory measurement
    gc.collect()
    initial_mem = psutil.virtual_memory().used / (1024**3)
    times_donate = []
    
    # For donate_argnames, we need to recreate params for each call
    for i in range(10):
        # Recreate params for each iteration since they get donated
        params_copy = {
            "weights": random.normal(key, shape=(2000, 2000)).astype(jnp.bfloat16),
            "bias": random.normal(key, shape=(2000,)).astype(jnp.bfloat16)
        }
        start_time = time.perf_counter()
        result = jit_donate_func(params_copy, input_data)
        times_donate.append(time.perf_counter() - start_time)
        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/10 runs with donate")
    
    peak_mem_donate = psutil.virtual_memory().used / (1024**3)
    
    # Calculate statistics
    avg_time_no_donate = sum(times_no_donate) / len(times_no_donate)
    avg_time_donate = sum(times_donate) / len(times_donate)
    
    print(f"\nResults:")
    print(f"Average time without donate: {avg_time_no_donate:.6f} seconds")
    print(f"Average time with donate: {avg_time_donate:.6f} seconds")
    print(f"Speedup: {avg_time_no_donate / avg_time_donate:.2f}x")
    print(f"Peak memory without donate: {peak_mem_no_donate:.2f} GB")
    print(f"Peak memory with donate: {peak_mem_donate:.2f} GB")
    print(f"Memory reduction: {((peak_mem_no_donate - peak_mem_donate) / peak_mem_no_donate * 100):.1f}%")
    
    return avg_time_no_donate / avg_time_donate

if __name__ == "__main__":
    print("=== Testing OPTIMIZATION 2: JIT with donate_argnames ===\n")
    
    # Test correctness and basic speedup
    correctness_passed, speedup1 = test_jit_donate_argnames()
    
    # Test memory efficiency
    speedup2 = test_memory_efficiency()
    
    print(f"\n=== SUMMARY ===")
    print(f"Correctness: {'✅ PASS' if correctness_passed else '❌ FAIL'}")
    print(f"Basic speedup: {speedup1:.2f}x")
    print(f"Memory efficiency speedup: {speedup2:.2f}x")
    
    if correctness_passed and speedup1 > 1.0 and speedup2 > 1.0:
        print("🎉 OPTIMIZATION 2 SUCCESSFUL!")
        print(f"Average speedup: {((speedup1 + speedup2) / 2):.2f}x")
    else:
        print("⚠️  OPTIMIZATION 2 NEEDS ATTENTION") 