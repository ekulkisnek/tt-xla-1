#!/usr/bin/env python3
"""
Simple test of ParallelDense in isolation
"""
import os
import jax
import jax.numpy as jnp
import flax.linen as nn

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

from q25j7_tensor_parallel_fixed import setup_device_mesh, ParallelDense

def test_parallel_dense():
    """Test ParallelDense with simple inputs"""
    
    # Setup
    mesh = setup_device_mesh()
    
    # Create test input
    rng = jax.random.PRNGKey(0)
    input_data = jnp.ones((2, 10, 64), dtype=jnp.float32)
    
    print("=== TESTING PARALLEL DENSE ===")
    print(f"Input shape: {input_data.shape}")
    print(f"Expected output shape: (2, 10, 128)")
    
    # Test regular Dense first
    print("\n--- Regular Dense ---")
    regular_layer = nn.Dense(features=128, dtype=jnp.float32)
    regular_params = regular_layer.init(rng, input_data)
    regular_output = regular_layer.apply(regular_params, input_data)
    
    print(f"Regular output shape: {regular_output.shape}")
    print(f"Regular output min/max: {jnp.min(regular_output):.4f}, {jnp.max(regular_output):.4f}")
    print(f"Regular output mean/std: {jnp.mean(regular_output):.4f}, {jnp.std(regular_output):.4f}")
    
    # Test ParallelDense
    print("\n--- Parallel Dense ---")
    parallel_layer = ParallelDense(features=128, dtype=jnp.float32)
    parallel_params = parallel_layer.init(rng, input_data)
    
    print(f"Parallel kernel shape: {parallel_params['params']['kernel'].shape}")
    
    with mesh:
        parallel_output = parallel_layer.apply(parallel_params, input_data)
    
    print(f"Parallel output shape: {parallel_output.shape}")
    print(f"Parallel output min/max: {jnp.min(parallel_output):.4f}, {jnp.max(parallel_output):.4f}")
    print(f"Parallel output mean/std: {jnp.mean(parallel_output):.4f}, {jnp.std(parallel_output):.4f}")
    
    # Compare outputs
    print("\n--- Comparison ---")
    if parallel_output.shape == regular_output.shape:
        print("✅ Shapes match!")
    else:
        print(f"❌ Shape mismatch: {regular_output.shape} vs {parallel_output.shape}")
    
    # Check if outputs are in similar ranges
    regular_range = jnp.max(regular_output) - jnp.min(regular_output)
    parallel_range = jnp.max(parallel_output) - jnp.min(parallel_output)
    print(f"Regular range: {regular_range:.4f}")
    print(f"Parallel range: {parallel_range:.4f}")
    
    if abs(regular_range - parallel_range) < 1.0:
        print("✅ Output ranges are similar!")
    else:
        print("❌ Output ranges differ significantly!")
    
    return parallel_output

if __name__ == "__main__":
    test_parallel_dense() 