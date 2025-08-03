#!/usr/bin/env python3
"""
Test script to compare original vs optimized version for OPTIMIZATION 1: Precomputed Rotary Embeddings
"""
import os
import sys
import time
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

# Add the current directory to path to import the modules
sys.path.append('.')

# Import both versions
import importlib.util

# Load original module
spec1 = importlib.util.spec_from_file_location("q25j7fast1", "q25j7fast1")
q25j7fast1 = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(q25j7fast1)
OriginalModel = q25j7fast1.Qwen25ForCausalLM

# Load optimized module
spec2 = importlib.util.spec_from_file_location("q25j7fast_opt1", "q25j7fast_opt1.py")
q25j7fast_opt1 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(q25j7fast_opt1)
OptimizedModel = q25j7fast_opt1.Qwen25ForCausalLM

def test_rotary_embeddings():
    """Test that both versions produce the same results."""
    print("Testing rotary embeddings optimization...")
    
    # Create a simple config for testing
    config = {
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_hidden_layers": 2,
        "vocab_size": 1000,
        "rope_theta": 1000000.0,
        "max_sequence_length": 1024,
        "rms_norm_eps": 1e-6
    }
    
    # Create models
    original_model = OriginalModel(config=config, dtype=jnp.bfloat16)
    optimized_model = OptimizedModel(config=config, dtype=jnp.bfloat16)
    
    # Create dummy input
    input_ids = jnp.array([[1, 2, 3, 4, 5]])
    position_ids = jnp.array([[0, 1, 2, 3, 4]])
    attention_mask = jnp.ones((1, 1, 5, 5), dtype=jnp.bfloat16)
    
    # Initialize parameters
    key = jax.random.PRNGKey(0)
    original_params = original_model.init(key, input_ids, attention_mask, position_ids)
    optimized_params = optimized_model.init(key, input_ids, attention_mask, position_ids)
    
    # Test forward pass
    print("Running forward pass comparison...")
    
    # Original version
    start_time = time.perf_counter()
    original_output = original_model.apply(original_params, input_ids, attention_mask, position_ids)
    original_time = time.perf_counter() - start_time
    
    # Optimized version
    start_time = time.perf_counter()
    optimized_output = optimized_model.apply(optimized_params, input_ids, attention_mask, position_ids)
    optimized_time = time.perf_counter() - start_time
    
    # Compare outputs
    original_logits = original_output["logits"]
    optimized_logits = optimized_output["logits"]
    
    # Check if outputs are close (allowing for small numerical differences)
    diff = jnp.abs(original_logits - optimized_logits)
    max_diff = jnp.max(diff)
    mean_diff = jnp.mean(diff)
    
    print(f"Original time: {original_time:.4f} seconds")
    print(f"Optimized time: {optimized_time:.4f} seconds")
    print(f"Speedup: {original_time / optimized_time:.2f}x")
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    
    # Check if differences are acceptable (should be very small due to numerical precision)
    if max_diff < 1e-3:
        print("✅ PASS: Outputs are sufficiently close")
        return True
    else:
        print("❌ FAIL: Outputs differ too much")
        return False

def test_speed_improvement():
    """Test speed improvement with longer sequences."""
    print("\nTesting speed improvement with longer sequences...")
    
    config = {
        "hidden_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "num_hidden_layers": 4,
        "vocab_size": 1000,
        "rope_theta": 1000000.0,
        "max_sequence_length": 2048,
        "rms_norm_eps": 1e-6
    }
    
    # Create models
    original_model = OriginalModel(config=config, dtype=jnp.bfloat16)
    optimized_model = OptimizedModel(config=config, dtype=jnp.bfloat16)
    
    # Create longer input
    seq_len = 512
    input_ids = jnp.array([[i for i in range(seq_len)]])
    position_ids = jnp.array([[i for i in range(seq_len)]])
    attention_mask = jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bfloat16)
    
    # Initialize parameters
    key = jax.random.PRNGKey(0)
    original_params = original_model.init(key, input_ids, attention_mask, position_ids)
    optimized_params = optimized_model.init(key, input_ids, attention_mask, position_ids)
    
    # Warm up
    print("Warming up...")
    for _ in range(3):
        _ = original_model.apply(original_params, input_ids, attention_mask, position_ids)
        _ = optimized_model.apply(optimized_params, input_ids, attention_mask, position_ids)
    
    # Benchmark original
    print("Benchmarking original version...")
    times = []
    for _ in range(5):
        start_time = time.perf_counter()
        _ = original_model.apply(original_params, input_ids, attention_mask, position_ids)
        times.append(time.perf_counter() - start_time)
    original_avg = sum(times) / len(times)
    
    # Benchmark optimized
    print("Benchmarking optimized version...")
    times = []
    for _ in range(5):
        start_time = time.perf_counter()
        _ = optimized_model.apply(optimized_params, input_ids, attention_mask, position_ids)
        times.append(time.perf_counter() - start_time)
    optimized_avg = sum(times) / len(times)
    
    print(f"Original average time: {original_avg:.4f} seconds")
    print(f"Optimized average time: {optimized_avg:.4f} seconds")
    print(f"Speedup: {original_avg / optimized_avg:.2f}x")
    
    return original_avg / optimized_avg

if __name__ == "__main__":
    print("=== Testing OPTIMIZATION 1: Precomputed Rotary Embeddings ===\n")
    
    # Test correctness
    correctness_passed = test_rotary_embeddings()
    
    # Test speed improvement
    speedup = test_speed_improvement()
    
    print(f"\n=== SUMMARY ===")
    print(f"Correctness: {'✅ PASS' if correctness_passed else '❌ FAIL'}")
    print(f"Speedup: {speedup:.2f}x")
    
    if correctness_passed and speedup > 1.0:
        print("🎉 OPTIMIZATION 1 SUCCESSFUL!")
    else:
        print("⚠️  OPTIMIZATION 1 NEEDS ATTENTION") 