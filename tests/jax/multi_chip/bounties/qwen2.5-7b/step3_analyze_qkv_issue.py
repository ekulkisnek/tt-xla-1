#!/usr/bin/env python3
"""
Step 3: Deep analysis of why Q/K/V ParallelDense breaks Llama attention
"""
import os
import jax
import jax.numpy as jnp
import json
import flax.linen as nn
from typing import Dict, Any, Optional, Union

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import q25j7_tensor_parallel_fixed
from q25j7_tensor_parallel_fixed import ParallelDense, setup_device_mesh
from transformers import AutoTokenizer

def debug_tensor_sharding(tensor, name):
    """Debug helper to show tensor properties"""
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Sharding: {tensor.sharding}")
    print(f"  Min/Max: {float(jnp.min(tensor)):.4f}, {float(jnp.max(tensor)):.4f}")
    print(f"  Mean/Std: {float(jnp.mean(tensor)):.4f}, {float(jnp.std(tensor)):.4f}")
    print()

class DebugParallelDense(ParallelDense):
    """ParallelDense with debug output to see what's happening"""
    
    def __call__(self, x):
        print(f"\n--- ParallelDense.{self.name} Input ---")
        debug_tensor_sharding(x, "Input")
        
        # Call parent implementation
        result = super().__call__(x)
        
        print(f"--- ParallelDense.{self.name} Output ---")
        debug_tensor_sharding(result, "Output")
        
        return result

def analyze_qkv_parallelization_issue():
    """Analyze exactly what happens when Q/K/V are parallelized"""
    
    mesh = setup_device_mesh()
    q25j7_tensor_parallel_fixed.mesh = mesh
    
    with open("weights/config.json") as f:
        config = json.load(f)
    
    print("=== STEP 3: ANALYZING Q/K/V PARALLELIZATION ISSUE ===\n")
    print("Goal: Understand exactly why Q/K/V ParallelDense breaks attention")
    
    # Test dimensions
    batch, seq = 1, 8
    hidden_size = config["hidden_size"]  # 3584
    num_heads = config["num_attention_heads"]  # 28  
    num_kv_heads = config["num_key_value_heads"]  # 4
    head_dim = hidden_size // num_heads  # 128
    
    print(f"Model dimensions:")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_heads: {num_heads}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  Q projection output: {num_heads * head_dim} = {hidden_size}")
    print(f"  K/V projection output: {num_kv_heads * head_dim} = {num_kv_heads * head_dim}")
    
    # Create test input
    rng = jax.random.PRNGKey(42)
    hidden_states = jnp.ones((batch, seq, hidden_size), dtype=jnp.bfloat16)
    
    print(f"\nTest input:")
    debug_tensor_sharding(hidden_states, "hidden_states")
    
    print("=" * 60)
    print("COMPARISON 1: Regular Dense vs ParallelDense for Q projection")
    print("=" * 60)
    
    # Test 1: Compare regular Dense vs ParallelDense for Q projection
    print("\n--- Creating Q projections ---")
    
    # Regular Dense (known to work)
    regular_q = nn.Dense(
        hidden_size,  # Q projection: 3584 -> 3584
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        use_bias=True,
        name="regular_q"
    )
    
    # ParallelDense (breaks attention)
    parallel_q = DebugParallelDense(
        hidden_size,  # Q projection: 3584 -> 3584
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        use_bias=True,
        name="parallel_q"
    )
    
    # Initialize both with same random weights
    print("\n--- Initializing projections ---")
    with mesh:
        regular_params = regular_q.init(rng, hidden_states)
        parallel_params = parallel_q.init(rng, hidden_states)
    
    print("✅ Both projections initialized")
    
    # Copy weights from regular to parallel to ensure identical computation
    print("\n--- Copying weights for fair comparison ---")
    
    # Extract regular weights
    regular_kernel = regular_params['params']['kernel']
    regular_bias = regular_params['params']['bias'] 
    
    print(f"Regular kernel shape: {regular_kernel.shape}")
    print(f"Regular bias shape: {regular_bias.shape}")
    
    # Set parallel weights to be identical
    parallel_params['params']['kernel'] = regular_kernel
    parallel_params['params']['bias'] = regular_bias
    
    print("✅ Weights copied - now projections should be mathematically identical")
    
    # Test forward pass
    print("\n--- Forward pass comparison ---")
    
    try:
        with mesh:
            print("Regular Dense forward pass:")
            regular_output = regular_q.apply(regular_params, hidden_states)
            debug_tensor_sharding(regular_output, "Regular Q output")
            
            print("ParallelDense forward pass:")
            parallel_output = parallel_q.apply(parallel_params, hidden_states)
            debug_tensor_sharding(parallel_output, "Parallel Q output")
        
        # Compare outputs
        print("--- Comparing outputs ---")
        diff = jnp.abs(regular_output - parallel_output)
        max_diff = float(jnp.max(diff))
        mean_diff = float(jnp.mean(diff))
        
        print(f"Max difference: {max_diff}")
        print(f"Mean difference: {mean_diff}")
        
        if max_diff < 1e-5:
            print("✅ Outputs are essentially identical")
        else:
            print("❌ Outputs differ significantly")
            print("This suggests the issue is in the ParallelDense implementation itself")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("COMPARISON 2: Sharding behavior analysis")
    print("=" * 60)
    
    # Test 2: Analyze what happens to sharding in attention computation
    print("\n--- Analyzing tensor sharding patterns ---")
    
    # Create a minimal attention computation to see where sharding breaks
    def _split_heads(hidden_states, num_heads):
        """Split hidden states into attention heads"""
        head_dim = hidden_states.shape[-1] // num_heads
        return hidden_states.reshape(
            hidden_states.shape[:2] + (num_heads, head_dim)
        )
    
    try:
        with mesh:
            # Get Q output from ParallelDense
            q_output = parallel_q.apply(parallel_params, hidden_states)
            debug_tensor_sharding(q_output, "Q before split_heads")
            
            # Reshape to attention heads
            q_heads = _split_heads(q_output, num_heads)
            debug_tensor_sharding(q_heads, "Q after split_heads")
            
            print("--- Analyzing head dimension distribution ---")
            print(f"Expected Q heads shape: ({batch}, {seq}, {num_heads}, {head_dim})")
            print(f"Actual Q heads shape: {q_heads.shape}")
            
            if q_heads.shape == (batch, seq, num_heads, head_dim):
                print("✅ Q head reshaping works correctly")
            else:
                print("❌ Q head reshaping failed - shape mismatch")
                
            # Check per-head statistics
            print("\n--- Per-head statistics ---")
            for i in range(min(4, num_heads)):  # Check first 4 heads
                head_data = q_heads[:, :, i, :]
                print(f"Head {i}: mean={float(jnp.mean(head_data)):.4f}, std={float(jnp.std(head_data)):.4f}")
            
            # Check if heads are identical (which would indicate a problem)
            head_0 = q_heads[:, :, 0, :]
            head_1 = q_heads[:, :, 1, :]
            head_diff = float(jnp.max(jnp.abs(head_0 - head_1)))
            
            if head_diff < 1e-6:
                print("❌ CRITICAL: All heads are identical! This indicates incorrect sharding.")
                print("The ParallelDense is not producing diverse head outputs.")
            else:
                print(f"✅ Heads are diverse (max diff between head 0 and 1: {head_diff})")
                
    except Exception as e:
        print(f"❌ Sharding analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("COMPARISON 3: K/V projection analysis (GQA)")
    print("=" * 60)
    
    # Test 3: Specific issue with K/V projections in GQA
    print("\n--- Analyzing K/V projections (GQA) ---")
    
    # K/V projections have fewer heads (4 vs 28)
    kv_output_dim = num_kv_heads * head_dim  # 4 * 128 = 512
    
    print(f"K/V projection dimensions:")
    print(f"  Input: {hidden_size}")  
    print(f"  Output: {kv_output_dim}")
    print(f"  Ratio: {hidden_size / kv_output_dim}")
    
    parallel_k = DebugParallelDense(
        kv_output_dim,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        use_bias=True,
        name="parallel_k"
    )
    
    try:
        with mesh:
            k_params = parallel_k.init(rng, hidden_states)
            k_output = parallel_k.apply(k_params, hidden_states)
            
            debug_tensor_sharding(k_output, "K projection output")
            
            # Reshape K to heads
            k_heads = _split_heads(k_output, num_kv_heads)
            debug_tensor_sharding(k_heads, "K after split_heads")
            
            print(f"Expected K heads shape: ({batch}, {seq}, {num_kv_heads}, {head_dim})")
            print(f"Actual K heads shape: {k_heads.shape}")
            
            # Check if the smaller output dimension causes sharding issues
            devices_per_dim = 4  # We have 4 devices
            if kv_output_dim % devices_per_dim != 0:
                print(f"⚠️ WARNING: K/V output dim ({kv_output_dim}) not divisible by devices ({devices_per_dim})")
                print("This could cause uneven sharding and collective communication issues")
            else:
                print(f"✅ K/V output dim ({kv_output_dim}) is divisible by devices ({devices_per_dim})")
            
    except Exception as e:
        print(f"❌ K/V analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return
        
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    print("\n🔍 SUMMARY OF FINDINGS:")
    print("1. ✅ Llama attention flow is compatible with Qwen")
    print("2. ✅ O-projection parallelization works perfectly")  
    print("3. ❌ Q/K/V projection parallelization breaks attention")
    print("4. 🔍 Analysis above shows exactly WHERE the issue occurs")
    
    print("\n📋 NEXT STEPS:")
    print("1. Check if heads become identical (sharding issue)")
    print("2. Check if GQA dimensions cause uneven sharding")
    print("3. Check if attention computation expects different sharding pattern")
    print("4. Investigate Llama's exact sharding specifications for Q/K/V")

if __name__ == "__main__":
    analyze_qkv_parallelization_issue()