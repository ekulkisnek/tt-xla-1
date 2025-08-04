#!/usr/bin/env python3
"""
Test nn.Dense vs ParallelDense with IDENTICAL weights to find the exact issue
"""
import os
import jax
import jax.numpy as jnp
import flax.linen as nn

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import q25j7_tensor_parallel_fixed
from q25j7_tensor_parallel_fixed import ParallelDense, setup_device_mesh

def debug_identical_weights():
    """Test Dense vs ParallelDense with exactly the same weights"""
    
    # Setup
    mesh = setup_device_mesh()
    q25j7_tensor_parallel_fixed.mesh = mesh
    
    print("=== TESTING DENSE vs PARALLELDENSE WITH IDENTICAL WEIGHTS ===\n")
    
    # Test parameters - use EXACT Qwen dimensions
    input_dim = 3584
    output_dim = 3584  # Start with Q projection dimensions
    batch, seq = 2, 8
    
    print(f"Input: ({batch}, {seq}, {input_dim})")
    print(f"Output: ({batch}, {seq}, {output_dim})")
    print(f"Devices: {mesh.shape['mp']}")
    
    # Create test input
    rng = jax.random.PRNGKey(42)
    input_data = jax.random.normal(rng, (batch, seq, input_dim), dtype=jnp.float32)
    
    print(f"Input range: {float(jnp.min(input_data)):.4f} to {float(jnp.max(input_data)):.4f}")
    
    # Create SHARED weights
    kernel_key = jax.random.PRNGKey(123)
    shared_kernel = jax.random.normal(kernel_key, (input_dim, output_dim), dtype=jnp.float32)
    
    print(f"Shared kernel shape: {shared_kernel.shape}")
    print(f"Shared kernel range: {float(jnp.min(shared_kernel)):.4f} to {float(jnp.max(shared_kernel)):.4f}")
    
    # Test 1: Regular Dense
    print("\n--- REGULAR DENSE ---")
    dense_layer = nn.Dense(output_dim, dtype=jnp.float32)
    dense_params = dense_layer.init(rng, input_data)
    # Replace with shared kernel
    dense_params["params"]["kernel"] = shared_kernel
    
    dense_output = dense_layer.apply(dense_params, input_data)
    print(f"Dense output shape: {dense_output.shape}")
    print(f"Dense output range: {float(jnp.min(dense_output)):.4f} to {float(jnp.max(dense_output)):.4f}")
    print(f"Dense output mean: {float(jnp.mean(dense_output)):.6f}")
    print(f"Dense output std: {float(jnp.std(dense_output)):.6f}")
    
    # Test 2: ParallelDense
    print("\n--- PARALLEL DENSE ---")
    parallel_layer = ParallelDense(output_dim, dtype=jnp.float32, param_dtype=jnp.float32)
    parallel_params = parallel_layer.init(rng, input_data)
    # Replace with SAME shared kernel
    parallel_params["params"]["kernel"] = shared_kernel
    
    with mesh:
        parallel_output = parallel_layer.apply(parallel_params, input_data)
    
    print(f"Parallel output shape: {parallel_output.shape}")
    print(f"Parallel output range: {float(jnp.min(parallel_output)):.4f} to {float(jnp.max(parallel_output)):.4f}")
    print(f"Parallel output mean: {float(jnp.mean(parallel_output)):.6f}")
    print(f"Parallel output std: {float(jnp.std(parallel_output)):.6f}")
    
    # Compare outputs element by element
    print("\n--- DETAILED COMPARISON ---")
    if dense_output.shape == parallel_output.shape:
        print("✅ Shapes match")
        
        # Element-wise comparison
        abs_diff = jnp.abs(dense_output - parallel_output)
        max_diff = float(jnp.max(abs_diff))
        mean_diff = float(jnp.mean(abs_diff))
        
        print(f"Max absolute difference: {max_diff:.10f}")
        print(f"Mean absolute difference: {mean_diff:.10f}")
        
        # Relative difference
        rel_diff = max_diff / float(jnp.max(jnp.abs(dense_output)))
        print(f"Max relative difference: {rel_diff:.10f} ({rel_diff*100:.8f}%)")
        
        if max_diff < 1e-6:
            print("✅ OUTPUTS ARE VIRTUALLY IDENTICAL!")
            return True
        elif max_diff < 1e-3:
            print("⚠️ Small numerical differences")
        else:
            print("❌ SIGNIFICANT DIFFERENCES FOUND!")
            
            # Show where the differences occur
            print(f"\nDifference statistics:")
            print(f"  Differences > 1e-6: {jnp.sum(abs_diff > 1e-6)}")
            print(f"  Differences > 1e-3: {jnp.sum(abs_diff > 1e-3)}")
            print(f"  Differences > 1e-1: {jnp.sum(abs_diff > 1e-1)}")
            
            # Find the location of max difference
            max_idx = jnp.unravel_index(jnp.argmax(abs_diff), abs_diff.shape)
            print(f"\nMax difference at position {max_idx}:")
            print(f"  Dense value: {float(dense_output[max_idx]):.10f}")
            print(f"  Parallel value: {float(parallel_output[max_idx]):.10f}")
            print(f"  Difference: {float(abs_diff[max_idx]):.10f}")
            
            # Show first few values
            print(f"\nFirst 5 output values:")
            print("Dense:   ", [f"{float(x):.6f}" for x in dense_output.flatten()[:5]])
            print("Parallel:", [f"{float(x):.6f}" for x in parallel_output.flatten()[:5]])
            
            return False
    else:
        print(f"❌ Shape mismatch: {dense_output.shape} vs {parallel_output.shape}")
        return False

def test_different_dimensions():
    """Test various dimensions to see where the issue occurs"""
    
    print("\n=== TESTING VARIOUS DIMENSIONS ===\n")
    
    # Test different output dimensions
    test_cases = [
        (3584, 3584, "Q projection"),      # Full size
        (3584, 512, "K/V projection"),    # GQA size
        (3584, 896, "Different size"),    # Another size
        (512, 512, "Small square"),       # Small size
    ]
    
    for input_dim, output_dim, description in test_cases:
        print(f"--- {description}: {input_dim} → {output_dim} ---")
        
        # Quick test
        mesh = setup_device_mesh()
        q25j7_tensor_parallel_fixed.mesh = mesh
        
        rng = jax.random.PRNGKey(42)
        input_data = jax.random.normal(rng, (1, 1, input_dim), dtype=jnp.float32)
        shared_kernel = jax.random.normal(jax.random.PRNGKey(123), (input_dim, output_dim), dtype=jnp.float32)
        
        # Dense
        dense_layer = nn.Dense(output_dim, dtype=jnp.float32)
        dense_params = dense_layer.init(rng, input_data)
        dense_params["params"]["kernel"] = shared_kernel
        dense_output = dense_layer.apply(dense_params, input_data)
        
        # Parallel
        parallel_layer = ParallelDense(output_dim, dtype=jnp.float32, param_dtype=jnp.float32)
        parallel_params = parallel_layer.init(rng, input_data)
        parallel_params["params"]["kernel"] = shared_kernel
        
        try:
            with mesh:
                parallel_output = parallel_layer.apply(parallel_params, input_data)
            
            max_diff = float(jnp.max(jnp.abs(dense_output - parallel_output)))
            
            if max_diff < 1e-6:
                print(f"✅ {description}: identical (diff: {max_diff:.2e})")
            else:
                print(f"❌ {description}: differs (diff: {max_diff:.2e})")
                
        except Exception as e:
            print(f"❌ {description}: failed ({e})")

if __name__ == "__main__":
    success = debug_identical_weights()
    if not success:
        print("\n" + "="*50)
        print("FOUND THE ISSUE!")
        print("ParallelDense produces different outputs than nn.Dense")
        print("even with identical weights!")
        print("="*50)
    
    test_different_dimensions()