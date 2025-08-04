#!/usr/bin/env python3
"""
Debug the actual parameter structure after loading
"""
import os
import jax
import jax.numpy as jnp
import json

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import q25j7_tensor_parallel_fixed
from q25j7_tensor_parallel_fixed import setup_device_mesh, load_params, Qwen25ForCausalLM

def debug_parameter_structure():
    """Debug the actual parameter structure after loading"""
    
    # Setup
    mesh = setup_device_mesh()
    q25j7_tensor_parallel_fixed.mesh = mesh
    
    with open("weights/config.json") as f:
        config = json.load(f)
    
    print("=== DEBUGGING PARAMETER STRUCTURE AFTER LOADING ===\n")
    
    # Create the current fully parallel model  
    model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    
    # Initialize model to see expected structure
    rng = jax.random.PRNGKey(42)
    input_ids = jnp.ones((1, 8), dtype=jnp.int32)
    init_params = model.init(rng, input_ids)
    
    print("--- EXPECTED PARAMETER STRUCTURE (after init) ---")
    def print_param_structure(params, prefix=""):
        for key, value in params.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}/")
                print_param_structure(value, prefix + "  ")
            else:
                if hasattr(value, 'shape'):
                    print(f"{prefix}{key}: {value.shape}")
                else:
                    print(f"{prefix}{key}: {type(value)}")
    
    print_param_structure(init_params)
    
    # Try to load weights
    print("\n--- ATTEMPTING TO LOAD WEIGHTS ---")
    try:
        loaded_params = load_params(model, "weights", jnp.bfloat16)
        print("✅ Weight loading succeeded!")
        
        print("\n--- LOADED PARAMETER STRUCTURE ---")
        print_param_structure(loaded_params)
        
        # Check specific attention parameters
        print("\n--- ATTENTION PARAMETER DETAILS ---")
        attn_params = loaded_params['params']['layers_0']['self_attn']
        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if proj in attn_params:
                print(f"{proj}:")
                for param_name, param_value in attn_params[proj].items():
                    print(f"  {param_name}: {param_value.shape}")
            else:
                print(f"{proj}: MISSING!")
        
        return True
        
    except Exception as e:
        print(f"❌ Weight loading failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Check what the load_params function is actually looking for
        print("\n--- CHECKING WHAT WEIGHTS ARE AVAILABLE ---")
        
        # Let's see what weight files contain
        try:
            import safetensors
            index_file = "weights/model.safetensors.index.json"
            with open(index_file) as f:
                index_data = json.load(f)
            
            print("Available weight keys (first 20):")
            weight_map = index_data["weight_map"]
            keys = list(weight_map.keys())[:20]
            for key in keys:
                print(f"  {key}")
            
            # Look specifically for attention weights
            print("\nAttention-related weight keys:")
            attention_keys = [k for k in weight_map.keys() if 'self_attn' in k and 'layers.0' in k]
            for key in attention_keys:
                print(f"  {key}")
                
        except Exception as e2:
            print(f"Could not check weight index: {e2}")
        
        return False

def debug_scope_issue():
    """Debug the scope issue specifically"""
    
    print("\n=== DEBUGGING SCOPE ISSUE ===\n")
    
    # Setup
    mesh = setup_device_mesh()
    q25j7_tensor_parallel_fixed.mesh = mesh
    
    with open("weights/config.json") as f:
        config = json.load(f)
    
    # Create a model that matches the ORIGINAL working structure for comparison
    from test_hybrid_model import HybridQwen25ForCausalLM
    
    print("--- HYBRID MODEL (working) PARAMETER STRUCTURE ---")
    hybrid_model = HybridQwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    rng = jax.random.PRNGKey(42)
    input_ids = jnp.ones((1, 8), dtype=jnp.int32)
    hybrid_init_params = hybrid_model.init(rng, input_ids)
    
    print("Hybrid attention structure:")
    hybrid_attn = hybrid_init_params['params']['layers_0']['self_attn']
    for key in hybrid_attn.keys():
        print(f"  {key}: {type(hybrid_attn[key])}")
        if isinstance(hybrid_attn[key], dict):
            for subkey in hybrid_attn[key].keys():
                if hasattr(hybrid_attn[key][subkey], 'shape'):
                    print(f"    {subkey}: {hybrid_attn[key][subkey].shape}")
    
    # Try loading weights for hybrid model
    print("\n--- LOADING WEIGHTS FOR HYBRID MODEL ---")
    try:
        hybrid_loaded = load_params(hybrid_model, "weights", jnp.bfloat16)
        print("✅ Hybrid model weight loading succeeded!")
        
        print("Hybrid loaded attention structure:")
        hybrid_loaded_attn = hybrid_loaded['params']['layers_0']['self_attn']
        for key in hybrid_loaded_attn.keys():
            print(f"  {key}: {type(hybrid_loaded_attn[key])}")
            if isinstance(hybrid_loaded_attn[key], dict):
                for subkey in hybrid_loaded_attn[key].keys():
                    if hasattr(hybrid_loaded_attn[key][subkey], 'shape'):
                        print(f"    {subkey}: {hybrid_loaded_attn[key][subkey].shape}")
        
    except Exception as e:
        print(f"❌ Hybrid model weight loading failed: {e}")

if __name__ == "__main__":
    success = debug_parameter_structure()
    debug_scope_issue()
    
    if not success:
        print("\n🔍 The issue is in the parameter structure mismatch!")
        print("ParallelDense creates different parameter paths than nn.Dense")
        print("The weight loading function needs to be updated to handle this.")