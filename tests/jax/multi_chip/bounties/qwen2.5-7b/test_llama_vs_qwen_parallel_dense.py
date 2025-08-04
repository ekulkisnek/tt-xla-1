#!/usr/bin/env python3
"""
Test to compare Llama's ParallelDense vs Qwen's ParallelDense
"""
import os
import jax
import jax.numpy as jnp
import sys

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

# Add Llama path
sys.path.append('/root/731/tests/jax/multi_chip/bounties/Llama_3.1-8B')

from llama.model import ParallelDense as LlamaParallelDense, Mesh, P
from q25j7_tensor_parallel_fixed import ParallelDense as QwenParallelDense, setup_device_mesh

def test_parallel_dense_comparison():
    """Compare Llama vs Qwen ParallelDense implementations"""
    
    # Setup mesh
    mesh = setup_device_mesh()
    
    print("=== LLAMA vs QWEN PARALLEL DENSE COMPARISON ===\n")
    
    # Test parameters
    input_shape = (2, 10, 64)  # batch, seq, features
    output_features = 128
    
    # Create test input
    rng = jax.random.PRNGKey(0)
    input_data = jnp.ones(input_shape, dtype=jnp.float32)
    
    print(f"Input shape: {input_shape}")
    print(f"Output features: {output_features}")
    
    # Test Llama ParallelDense
    print("\n--- LLAMA PARALLEL DENSE ---")
    try:
        llama_layer = LlamaParallelDense(
            features=output_features,
            dtype=jnp.float32,
            param_dtype=jnp.float32
        )
        
        # Initialize
        llama_params = llama_layer.init(rng, input_data)
        print(f"Llama kernel shape: {llama_params['params']['kernel'].shape}")
        
        # Forward pass
        with mesh:
            llama_output = llama_layer.apply(llama_params, input_data)
        
        print(f"Llama output shape: {llama_output.shape}")
        print(f"Llama output min/max: {float(jnp.min(llama_output)):.4f}, {float(jnp.max(llama_output)):.4f}")
        print("✅ Llama ParallelDense works!")
        
    except Exception as e:
        print(f"❌ Llama ParallelDense failed: {e}")
        llama_output = None
    
    # Test Qwen ParallelDense
    print("\n--- QWEN PARALLEL DENSE ---")
    try:
        qwen_layer = QwenParallelDense(
            features=output_features,
            dtype=jnp.float32,
            param_dtype=jnp.float32
        )
        
        # Initialize with same RNG for comparison
        qwen_params = qwen_layer.init(rng, input_data)
        print(f"Qwen kernel shape: {qwen_params['params']['kernel'].shape}")
        
        # Forward pass
        with mesh:
            qwen_output = qwen_layer.apply(qwen_params, input_data)
        
        print(f"Qwen output shape: {qwen_output.shape}")
        print(f"Qwen output min/max: {float(jnp.min(qwen_output)):.4f}, {float(jnp.max(qwen_output)):.4f}")
        print("✅ Qwen ParallelDense works!")
        
    except Exception as e:
        print(f"❌ Qwen ParallelDense failed: {e}")
        qwen_output = None
    
    # Compare outputs if both work
    if llama_output is not None and qwen_output is not None:
        print("\n--- COMPARISON ---")
        if llama_output.shape == qwen_output.shape:
            print("✅ Shapes match!")
            
            # Compare ranges
            llama_range = float(jnp.max(llama_output) - jnp.min(llama_output))
            qwen_range = float(jnp.max(qwen_output) - jnp.min(qwen_output))
            print(f"Llama range: {llama_range:.4f}")
            print(f"Qwen range: {qwen_range:.4f}")
            
            if abs(llama_range - qwen_range) < 1.0:
                print("✅ Output ranges are similar!")
            else:
                print("❌ Output ranges differ significantly!")
                
        else:
            print(f"❌ Shape mismatch: Llama {llama_output.shape} vs Qwen {qwen_output.shape}")
    
    return llama_output, qwen_output

if __name__ == "__main__":
    test_parallel_dense_comparison()