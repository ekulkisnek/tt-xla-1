#!/usr/bin/env python3
"""
Debug the exact tensor flow in attention with parallel vs regular projections
"""
import os
import jax
import jax.numpy as jnp
import flax.linen as nn

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import q25j7_tensor_parallel_fixed
from q25j7_tensor_parallel_fixed import ParallelDense, setup_device_mesh

def debug_attention_tensor_flow():
    """Compare tensor flow with parallel vs regular projections"""
    
    # Setup
    mesh = setup_device_mesh()
    q25j7_tensor_parallel_fixed.mesh = mesh
    
    print("=== DEBUGGING ATTENTION TENSOR FLOW ===\n")
    
    # Qwen config values
    hidden_size = 3584
    num_heads = 28
    num_kv_heads = 4
    head_dim = hidden_size // num_heads  # 128
    kv_dim = num_kv_heads * head_dim     # 512
    
    print(f"hidden_size: {hidden_size}")
    print(f"num_heads: {num_heads}, num_kv_heads: {num_kv_heads}")
    print(f"head_dim: {head_dim}, kv_dim: {kv_dim}")
    
    # Test input
    batch, seq = 1, 8
    rng = jax.random.PRNGKey(42)
    input_data = jax.random.normal(rng, (batch, seq, hidden_size), dtype=jnp.float32)
    
    print(f"Input shape: {input_data.shape}")
    
    # Test 1: Regular Dense projections (working case)
    print("\n--- REGULAR DENSE PROJECTIONS ---")
    
    q_dense = nn.Dense(hidden_size, dtype=jnp.float32, name="q_proj")
    k_dense = nn.Dense(kv_dim, dtype=jnp.float32, name="k_proj")
    v_dense = nn.Dense(kv_dim, dtype=jnp.float32, name="v_proj")
    
    q_params = q_dense.init(rng, input_data)
    k_params = k_dense.init(rng, input_data)
    v_params = v_dense.init(rng, input_data)
    
    q_regular = q_dense.apply(q_params, input_data)
    k_regular = k_dense.apply(k_params, input_data)
    v_regular = v_dense.apply(v_params, input_data)
    
    print(f"Regular Q output: {q_regular.shape}")
    print(f"Regular K output: {k_regular.shape}")
    print(f"Regular V output: {v_regular.shape}")
    
    # Reshape for attention
    q_regular_reshaped = q_regular.reshape(batch, seq, num_heads, head_dim)
    k_regular_reshaped = k_regular.reshape(batch, seq, num_kv_heads, head_dim)
    v_regular_reshaped = v_regular.reshape(batch, seq, num_kv_heads, head_dim)
    
    print(f"Regular Q reshaped: {q_regular_reshaped.shape}")
    print(f"Regular K reshaped: {k_regular_reshaped.shape}")
    print(f"Regular V reshaped: {v_regular_reshaped.shape}")
    
    # Test 2: Parallel Dense projections (failing case)
    print("\n--- PARALLEL DENSE PROJECTIONS ---")
    
    q_parallel = ParallelDense(hidden_size, dtype=jnp.float32, param_dtype=jnp.float32, name="q_proj")
    k_parallel = ParallelDense(kv_dim, dtype=jnp.float32, param_dtype=jnp.float32, name="k_proj")
    v_parallel = ParallelDense(kv_dim, dtype=jnp.float32, param_dtype=jnp.float32, name="v_proj")
    
    q_par_params = q_parallel.init(rng, input_data)
    k_par_params = k_parallel.init(rng, input_data)
    v_par_params = v_parallel.init(rng, input_data)
    
    # Use same weights as regular for fair comparison
    q_par_params['params']['kernel'] = q_params['params']['kernel']
    k_par_params['params']['kernel'] = k_params['params']['kernel']
    v_par_params['params']['kernel'] = v_params['params']['kernel']
    
    with mesh:
        q_par = q_parallel.apply(q_par_params, input_data)
        k_par = k_parallel.apply(k_par_params, input_data)
        v_par = v_parallel.apply(v_par_params, input_data)
    
    print(f"Parallel Q output: {q_par.shape}")
    print(f"Parallel K output: {k_par.shape}")
    print(f"Parallel V output: {v_par.shape}")
    
    # Reshape for attention
    q_par_reshaped = q_par.reshape(batch, seq, num_heads, head_dim)
    k_par_reshaped = k_par.reshape(batch, seq, num_kv_heads, head_dim)
    v_par_reshaped = v_par.reshape(batch, seq, num_kv_heads, head_dim)
    
    print(f"Parallel Q reshaped: {q_par_reshaped.shape}")
    print(f"Parallel K reshaped: {k_par_reshaped.shape}")
    print(f"Parallel V reshaped: {v_par_reshaped.shape}")
    
    # Compare the actual tensor values
    print("\n--- TENSOR VALUE COMPARISON ---")
    
    q_diff = jnp.max(jnp.abs(q_regular - q_par))
    k_diff = jnp.max(jnp.abs(k_regular - k_par))
    v_diff = jnp.max(jnp.abs(v_regular - v_par))
    
    print(f"Max Q difference: {float(q_diff):.8f}")
    print(f"Max K difference: {float(k_diff):.8f}")
    print(f"Max V difference: {float(v_diff):.8f}")
    
    if q_diff < 1e-5 and k_diff < 1e-5 and v_diff < 1e-5:
        print("✅ Tensor values are nearly identical!")
        print("The issue is NOT in the projection outputs themselves")
        return True
    else:
        print("❌ Tensor values differ significantly!")
        print("The issue IS in the projection computation")
        
        # Show first few values for debugging
        if q_diff > 1e-5:
            print(f"\nQ tensor differences (first 5 elements):")
            print(f"Regular: {q_regular.flatten()[:5]}")
            print(f"Parallel: {q_par.flatten()[:5]}")
        
        return False

def debug_attention_sharding():
    """Debug if the issue is in tensor sharding patterns"""
    
    print("\n=== DEBUGGING ATTENTION SHARDING PATTERNS ===\n")
    
    # Check if the tensors have different sharding after parallel projections
    print("This would require more advanced JAX sharding inspection")
    print("But based on our previous tests, the values should be identical")
    print("So the issue is likely in subsequent operations, not the projections themselves")

if __name__ == "__main__":
    values_match = debug_attention_tensor_flow()
    debug_attention_sharding()
    
    if values_match:
        print("\n🔍 Since projection outputs are identical, the issue must be in:")
        print("1. How the tensors are sharded/distributed after projection")
        print("2. How subsequent attention operations handle sharded tensors")
        print("3. Some interaction between multiple parallel projections")
    else:
        print("\n🔍 The issue is in the projection computation itself")
        print("Need to debug why ParallelDense produces different values")