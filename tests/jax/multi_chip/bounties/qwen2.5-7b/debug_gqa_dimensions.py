#!/usr/bin/env python3
"""
Debug ParallelDense behavior with different GQA dimensions
"""
import os
import jax
import jax.numpy as jnp
import flax.linen as nn

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import q25j7_tensor_parallel_fixed
from q25j7_tensor_parallel_fixed import ParallelDense, setup_device_mesh

def debug_gqa_dimensions():
    """Debug how ParallelDense handles different output dimensions"""
    
    # Setup
    mesh = setup_device_mesh()
    q25j7_tensor_parallel_fixed.mesh = mesh
    
    print("=== DEBUGGING GQA DIMENSION HANDLING ===\n")
    
    # Test parameters (matching Qwen2.5-7B config)
    input_dim = 3584
    q_output_dim = 3584  # Full hidden size for Q
    kv_output_dim = 512  # Much smaller for K/V due to GQA (4 heads vs 28)
    
    print(f"Input dimension: {input_dim}")
    print(f"Q output dimension: {q_output_dim} (full)")
    print(f"K/V output dimension: {kv_output_dim} (GQA)")
    print(f"Parallel devices: {mesh.shape['mp']}")
    print(f"K/V dimension per device: {kv_output_dim // mesh.shape['mp']}")
    
    # Check if K/V dimensions are divisible by number of devices
    if kv_output_dim % mesh.shape['mp'] != 0:
        print(f"❌ PROBLEM: K/V dimension {kv_output_dim} is NOT divisible by {mesh.shape['mp']} devices!")
        print(f"   This will cause issues in tensor parallelism!")
    else:
        print(f"✅ K/V dimension {kv_output_dim} is divisible by {mesh.shape['mp']} devices")
    
    # Test input
    rng = jax.random.PRNGKey(0)
    input_shape = (2, 10, input_dim)
    test_input = jnp.ones(input_shape, dtype=jnp.float32)
    
    print(f"\nTest input shape: {input_shape}")
    
    # Test 1: Q projection (should work - large dimension)
    print("\n--- Q PROJECTION TEST (large dimension) ---")
    try:
        q_layer_parallel = ParallelDense(q_output_dim, dtype=jnp.float32, param_dtype=jnp.float32)
        q_layer_regular = nn.Dense(q_output_dim, dtype=jnp.float32)
        
        # Initialize both with same RNG
        q_params_parallel = q_layer_parallel.init(rng, test_input)
        q_params_regular = q_layer_regular.init(rng, test_input)
        
        print(f"Q Parallel kernel shape: {q_params_parallel['params']['kernel'].shape}")
        print(f"Q Regular kernel shape: {q_params_regular['params']['kernel'].shape}")
        
        # Forward pass
        with mesh:
            q_output_parallel = q_layer_parallel.apply(q_params_parallel, test_input)
        q_output_regular = q_layer_regular.apply(q_params_regular, test_input)
        
        print(f"Q Parallel output shape: {q_output_parallel.shape}")
        print(f"Q Regular output shape: {q_output_regular.shape}")
        
        # Compare outputs (they should be different due to different initialization)
        q_parallel_range = float(jnp.max(q_output_parallel) - jnp.min(q_output_parallel))
        q_regular_range = float(jnp.max(q_output_regular) - jnp.min(q_output_regular))
        
        print(f"Q Parallel range: {q_parallel_range:.4f}")
        print(f"Q Regular range: {q_regular_range:.4f}")
        
        if q_output_parallel.shape == q_output_regular.shape:
            print("✅ Q projection shapes match")
        else:
            print("❌ Q projection shapes differ")
            
    except Exception as e:
        print(f"❌ Q projection failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: K/V projection (problematic - small dimension)
    print("\n--- K/V PROJECTION TEST (small dimension) ---")
    try:
        kv_layer_parallel = ParallelDense(kv_output_dim, dtype=jnp.float32, param_dtype=jnp.float32)
        kv_layer_regular = nn.Dense(kv_output_dim, dtype=jnp.float32)
        
        # Initialize both with same RNG
        kv_params_parallel = kv_layer_parallel.init(rng, test_input)
        kv_params_regular = kv_layer_regular.init(rng, test_input)
        
        print(f"KV Parallel kernel shape: {kv_params_parallel['params']['kernel'].shape}")
        print(f"KV Regular kernel shape: {kv_params_regular['params']['kernel'].shape}")
        
        # Forward pass
        with mesh:
            kv_output_parallel = kv_layer_parallel.apply(kv_params_parallel, test_input)
        kv_output_regular = kv_layer_regular.apply(kv_params_regular, test_input)
        
        print(f"KV Parallel output shape: {kv_output_parallel.shape}")
        print(f"KV Regular output shape: {kv_output_regular.shape}")
        
        # Compare outputs
        kv_parallel_range = float(jnp.max(kv_output_parallel) - jnp.min(kv_output_parallel))
        kv_regular_range = float(jnp.max(kv_output_regular) - jnp.min(kv_output_regular))
        
        print(f"KV Parallel range: {kv_parallel_range:.4f}")
        print(f"KV Regular range: {kv_regular_range:.4f}")
        
        if kv_output_parallel.shape == kv_output_regular.shape:
            print("✅ KV projection shapes match")
        else:
            print("❌ KV projection shapes differ")
            
        # Check if the actual values are similar ranges (they should be for correct parallelization)
        range_ratio = kv_parallel_range / kv_regular_range if kv_regular_range != 0 else float('inf')
        print(f"Range ratio (parallel/regular): {range_ratio:.4f}")
        
        if 0.5 < range_ratio < 2.0:
            print("✅ Ranges are similar - parallelization likely correct")
        else:
            print("❌ Ranges differ significantly - parallelization issue!")
            
    except Exception as e:
        print(f"❌ KV projection failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Check what happens with dimensions that don't divide evenly
    print("\n--- DIMENSION DIVISIBILITY TEST ---")
    test_dims = [512, 513, 514, 515, 516]  # Some divisible by 4, some not
    
    for dim in test_dims:
        divisible = dim % mesh.shape['mp'] == 0
        per_device = dim // mesh.shape['mp'] if divisible else dim / mesh.shape['mp']
        print(f"Dimension {dim}: {'✅' if divisible else '❌'} divisible by {mesh.shape['mp']}, {per_device:.1f} per device")

if __name__ == "__main__":
    debug_gqa_dimensions()