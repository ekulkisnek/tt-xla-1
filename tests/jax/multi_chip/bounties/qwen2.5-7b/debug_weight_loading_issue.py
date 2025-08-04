#!/usr/bin/env python3
"""
Debug if weight loading is corrupted for ParallelDense attention layers
"""
import os
import jax
import jax.numpy as jnp
import json
import flax.linen as nn

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import q25j7_tensor_parallel_fixed
from q25j7_tensor_parallel_fixed import ParallelDense, setup_device_mesh

def debug_weight_loading_issue():
    """Test if weight loading corrupts ParallelDense attention weights"""
    
    # Setup
    mesh = setup_device_mesh()
    q25j7_tensor_parallel_fixed.mesh = mesh
    
    with open("weights/config.json") as f:
        config = json.load(f)
    
    print("=== DEBUGGING WEIGHT LOADING FOR PARALLEL ATTENTION ===\n")
    
    # Create a simple test case: single Q projection layer
    input_dim = config["hidden_size"]  # 3584
    output_dim = config["hidden_size"] # 3584
    
    print(f"Testing Q projection: {input_dim} → {output_dim}")
    
    # Test input
    rng = jax.random.PRNGKey(42)
    batch, seq = 1, 1
    test_input = jnp.ones((batch, seq, input_dim), dtype=jnp.bfloat16)
    
    print(f"Test input shape: {test_input.shape}")
    
    # Test 1: nn.Dense with "loaded" weights (simulated)
    print("\n--- NN.DENSE WITH LOADED WEIGHTS ---")
    
    # Create some "loaded" weights (simulating the weight loading process)
    loaded_weight = jax.random.normal(jax.random.PRNGKey(123), (output_dim, input_dim), dtype=jnp.bfloat16)
    print(f"Loaded weight shape (HF format): {loaded_weight.shape}")
    
    # nn.Dense expects transposed weights
    dense_layer = nn.Dense(output_dim, dtype=jnp.bfloat16)
    dense_params = dense_layer.init(rng, test_input)
    dense_params["params"]["kernel"] = loaded_weight.T  # Transpose for Flax format
    
    dense_output = dense_layer.apply(dense_params, test_input)
    print(f"Dense output: {dense_output[0, 0, :5]}")  # First 5 values
    print(f"Dense output range: {float(jnp.min(dense_output)):.6f} to {float(jnp.max(dense_output)):.6f}")
    
    # Test 2: ParallelDense with SAME loaded weights 
    print("\n--- PARALLELDENSE WITH SAME LOADED WEIGHTS ---")
    
    parallel_layer = ParallelDense(output_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
    parallel_params = parallel_layer.init(rng, test_input)
    parallel_params["params"]["kernel"] = loaded_weight.T  # Same transposed weights
    
    with mesh:
        parallel_output = parallel_layer.apply(parallel_params, test_input)
    
    print(f"Parallel output: {parallel_output[0, 0, :5]}")  # First 5 values  
    print(f"Parallel output range: {float(jnp.min(parallel_output)):.6f} to {float(jnp.max(parallel_output)):.6f}")
    
    # Compare outputs
    print("\n--- COMPARISON ---")
    max_diff = float(jnp.max(jnp.abs(dense_output - parallel_output)))
    rel_diff = max_diff / float(jnp.max(jnp.abs(dense_output))) if float(jnp.max(jnp.abs(dense_output))) > 0 else 0
    
    print(f"Max absolute difference: {max_diff:.8f}")
    print(f"Relative difference: {rel_diff:.8f} ({rel_diff*100:.6f}%)")
    
    if max_diff < 1e-6:
        print("✅ Weight loading works correctly for ParallelDense")
        return True
    else:
        print("❌ Weight loading is corrupted for ParallelDense!")
        
        # Show the first few values side by side
        print("\nFirst 10 values comparison:")
        dense_vals = dense_output[0, 0, :10]
        parallel_vals = parallel_output[0, 0, :10]
        
        for i in range(10):
            d_val = float(dense_vals[i])
            p_val = float(parallel_vals[i])
            diff = abs(d_val - p_val)
            print(f"  {i}: Dense {d_val:.6f}, Parallel {p_val:.6f}, Diff {diff:.6f}")
        
        return False

def test_weight_loading_with_actual_loading_function():
    """Test with the actual weight loading function used in the model"""
    
    print("\n=== TESTING WITH ACTUAL WEIGHT LOADING ===\n")
    
    # Setup  
    mesh = setup_device_mesh()
    q25j7_tensor_parallel_fixed.mesh = mesh
    
    with open("weights/config.json") as f:
        config = json.load(f)
    
    # Create minimal models with just one attention layer
    class MinimalDenseModel(nn.Module):
        def setup(self):
            self.q_proj = nn.Dense(config["hidden_size"], dtype=jnp.bfloat16, name="q_proj")
        
        def __call__(self, x):
            return self.q_proj(x)
    
    class MinimalParallelModel(nn.Module):
        def setup(self):
            self.q_proj = ParallelDense(config["hidden_size"], dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name="q_proj")
        
        def __call__(self, x):
            return self.q_proj(x)
    
    # Test input
    rng = jax.random.PRNGKey(42)
    test_input = jnp.ones((1, 1, config["hidden_size"]), dtype=jnp.bfloat16)
    
    # Test Dense model with actual weight loading
    print("--- DENSE MODEL WITH ACTUAL WEIGHT LOADING ---")
    dense_model = MinimalDenseModel()
    dense_params = dense_model.init(rng, test_input)
    
    # Load actual weights (simplified version of load_params)
    try:
        from q25j7_tensor_parallel_fixed import load_params
        dense_params_loaded = load_params(dense_model, "weights", jnp.bfloat16)
        
        dense_output = dense_model.apply(dense_params_loaded, test_input)
        print(f"Dense with loaded weights: {dense_output[0, 0, :5]}")
        print("✅ Dense model loaded successfully")
        
    except Exception as e:
        print(f"❌ Dense model loading failed: {e}")
        return False
    
    # Test Parallel model with actual weight loading
    print("\n--- PARALLEL MODEL WITH ACTUAL WEIGHT LOADING ---")
    parallel_model = MinimalParallelModel()
    parallel_params = parallel_model.init(rng, test_input)
    
    try:
        parallel_params_loaded = load_params(parallel_model, "weights", jnp.bfloat16)
        
        with mesh:
            parallel_output = parallel_model.apply(parallel_params_loaded, test_input)
        
        print(f"Parallel with loaded weights: {parallel_output[0, 0, :5]}")
        print("✅ Parallel model loaded successfully")
        
        # Compare with dense output
        max_diff = float(jnp.max(jnp.abs(dense_output - parallel_output)))
        print(f"\nDifference vs Dense: {max_diff:.8f}")
        
        if max_diff < 1e-3:
            print("✅ Parallel loading produces similar results to Dense")
            return True
        else:
            print("❌ Parallel loading produces different results!")
            return False
        
    except Exception as e:
        print(f"❌ Parallel model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing weight loading for ParallelDense attention layers...\n")
    
    # Test 1: Simulated weight loading  
    success1 = debug_weight_loading_issue()
    
    # Test 2: Actual weight loading
    success2 = test_weight_loading_with_actual_loading_function()
    
    if success1 and success2:
        print("\n✅ WEIGHT LOADING IS NOT THE ISSUE")
        print("The problem must be elsewhere in the attention computation")
    else:
        print("\n❌ WEIGHT LOADING IS CORRUPTED!")
        print("This is likely the root cause of the prediction differences")