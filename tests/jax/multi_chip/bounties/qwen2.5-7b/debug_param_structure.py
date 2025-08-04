#!/usr/bin/env python3
"""
Debug parameter structure differences
"""
import os
import jax
import jax.numpy as jnp

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

from q25j7_tensor_parallel_fixed import ParallelDense as QwenParallelDense, setup_device_mesh

def debug_param_structure():
    """Debug parameter structure"""
    
    mesh = setup_device_mesh()
    
    print("=== DEBUGGING PARAMETER STRUCTURE ===\n")
    
    # Test parameters
    input_shape = (2, 10, 64)
    output_features = 128
    
    # Create test input
    rng = jax.random.PRNGKey(0)
    input_data = jnp.ones(input_shape, dtype=jnp.float32)
    
    # Test Qwen ParallelDense
    print("--- QWEN PARALLEL DENSE PARAMS ---")
    qwen_layer = QwenParallelDense(
        features=output_features,
        dtype=jnp.float32,
        param_dtype=jnp.float32
    )
    
    qwen_params = qwen_layer.init(rng, input_data)
    print(f"Qwen params keys: {list(qwen_params.keys())}")
    print(f"Qwen params['params'] keys: {list(qwen_params['params'].keys())}")
    
    def print_nested_structure(d, level=0):
        indent = "  " * level
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{indent}{key}/")
                print_nested_structure(value, level + 1)
            else:
                if hasattr(value, 'shape'):
                    print(f"{indent}{key}: {value.shape}")
                else:
                    print(f"{indent}{key}: {type(value)}")
    
    print("\nFull parameter structure:")
    print_nested_structure(qwen_params)

if __name__ == "__main__":
    debug_param_structure()