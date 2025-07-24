#!/usr/bin/env python3
"""
Phase 3.13: Debug Weight Loading Process
=======================================
Debug the exact weight loading process to see why transpose isn't working correctly.
"""

import sys
import os
import numpy as np
import json
from pathlib import Path
from safetensors import safe_open

def debug_weight_loading():
    """Debug the weight loading process step by step"""
    
    print("=" * 70)
    print("PHASE 3.13: DEBUG WEIGHT LOADING PROCESS")
    print("=" * 70)
    
    model_path = "../../weights"
    
    # Test specific keys that we know are problematic
    test_keys = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight", 
        "model.layers.0.self_attn.v_proj.weight"
    ]
    
    print("Testing safetensors loading with specific projection keys...")
    
    # Load from safetensors and debug step by step
    for file in os.listdir(model_path):
        if not file.endswith(".safetensors"):
            continue
            
        file_path = os.path.join(model_path, file)
        print(f"\n--- Processing {file} ---")
        
        with safe_open(file_path, framework="numpy") as f:
            available_keys = f.keys()
            
            for test_key in test_keys:
                if test_key in available_keys:
                    print(f"\n🎯 Found {test_key}")
                    
                    # Get original parameter
                    param = f.get_tensor(test_key)
                    print(f"Original shape: {param.shape}")
                    print(f"Original dtype: {param.dtype}")
                    
                    # Convert to numpy first to avoid bfloat16 issues
                    if hasattr(param, 'numpy'):
                        param_np = param.numpy()
                    else:
                        param_np = np.array(param)
                    print(f"Original sample: {param_np.flat[:5]}")
                    
                    # Test our path mapping
                    sys.path.append("../..")
                    from qwen_jax_inference import get_param_path, transpose_if_needed
                    
                    param_path = get_param_path(test_key)
                    print(f"Mapped path: {param_path}")
                    
                    # Test our dtype conversion  
                    import jax.numpy as jnp
                    
                    print(f"\n--- Testing dtype conversion ---")
                    original_dtype = param_np.dtype
                    target_dtype = jnp.bfloat16
                    
                    if original_dtype == np.float16 and target_dtype == jnp.bfloat16:
                        print("Applying direct conversion (our fix)")
                        converted = jnp.array(param_np, dtype=target_dtype)
                    else:
                        print("Using standard conversion")
                        converted = jnp.array(param_np, dtype=target_dtype)
                    
                    print(f"Converted shape: {converted.shape}")
                    print(f"Converted dtype: {converted.dtype}")
                    print(f"Converted sample: {converted.flat[:5]}")
                    
                    # Test transpose
                    print(f"\n--- Testing transpose ---")
                    print(f"Should transpose? {'weight' in test_key and 'proj' in test_key}")
                    
                    transposed = transpose_if_needed(test_key, converted)
                    print(f"After transpose shape: {transposed.shape}")
                    print(f"After transpose sample: {transposed.flat[:5]}")
                    
                    # Compare with PyTorch 
                    print(f"\n--- Comparing with PyTorch ---")
                    pytorch_weight = load_pytorch_weight(test_key)
                    if pytorch_weight is not None:
                        print(f"PyTorch shape: {pytorch_weight.shape}")
                        print(f"PyTorch sample: {pytorch_weight.flat[:5]}")
                        
                        # Check if our JAX weight matches PyTorch (accounting for transpose)
                        if pytorch_weight.shape == transposed.shape:
                            diff = np.max(np.abs(pytorch_weight - np.array(transposed)))
                            print(f"Max difference: {diff:.8e}")
                            if diff < 1e-6:
                                print("✅ Weights match!")
                            else:
                                print("❌ Weights differ significantly")
                        elif pytorch_weight.shape == converted.shape:
                            diff = np.max(np.abs(pytorch_weight - np.array(converted)))
                            print(f"Max difference (no transpose): {diff:.8e}")
                            print("🔍 Transpose may be incorrect")
                        else:
                            print(f"❌ Shape mismatch: PT {pytorch_weight.shape} vs JAX {transposed.shape}")
                    
                    print("-" * 50)
                    
    return True

def load_pytorch_weight(key):
    """Load the same weight from PyTorch for comparison"""
    try:
        from transformers import AutoModelForCausalLM
        import torch
        
        model_path = "../../weights"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Navigate to the specific parameter
        if key == "model.layers.0.self_attn.q_proj.weight":
            weight = model.model.layers[0].self_attn.q_proj.weight
        elif key == "model.layers.0.self_attn.k_proj.weight":
            weight = model.model.layers[0].self_attn.k_proj.weight
        elif key == "model.layers.0.self_attn.v_proj.weight":
            weight = model.model.layers[0].self_attn.v_proj.weight
        else:
            return None
            
        result = weight.detach().float().numpy()
        
        # Cleanup
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return result
        
    except Exception as e:
        print(f"Error loading PyTorch weight: {e}")
        return None

def main():
    """Main test function"""
    success = debug_weight_loading()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 