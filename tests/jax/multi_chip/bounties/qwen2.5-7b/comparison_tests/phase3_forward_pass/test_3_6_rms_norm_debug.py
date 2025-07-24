#!/usr/bin/env python3
"""
Phase 3.6: RMS Normalization Debug
=================================
Focused test to debug RMS normalization differences between PyTorch and JAX.
"""

import sys
import os
import numpy as np
import time
import gc
import json
import math
from pathlib import Path

def get_test_input():
    """Get test input for RMS norm analysis"""
    return "Hello"  # Token 9707 - confirmed identical embedding

def extract_pytorch_rms_norm_details(test_input):
    """Extract detailed RMS normalization components from PyTorch"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print("Loading PyTorch model for RMS norm analysis...")
        model_path = "../../weights"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model.eval()
        
        # Tokenize input
        inputs = tokenizer(test_input, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        print(f"PyTorch input: {test_input}")
        print(f"PyTorch tokens: {input_ids.tolist()}")
        
        # Extract embedding (confirmed identical)
        with torch.no_grad():
            embedding_output = model.model.embed_tokens(input_ids)
            
            # Get Layer 0 input layernorm
            layer_0 = model.model.layers[0]
            input_layernorm = layer_0.input_layernorm
            
            # Get the layernorm weight
            ln_weight = input_layernorm.weight
            
            # Manual RMS normalization computation
            x = embedding_output.float()
            
            # Step 1: Compute variance (mean of squares)
            variance = x.pow(2).mean(-1, keepdim=True)
            
            # Step 2: Add epsilon
            eps = input_layernorm.variance_epsilon if hasattr(input_layernorm, 'variance_epsilon') else 1e-6
            variance_with_eps = variance + eps
            
            # Step 3: Compute RMS (square root)
            rms = torch.sqrt(variance_with_eps)
            
            # Step 4: Normalize
            normalized = x / rms
            
            # Step 5: Scale
            output_manual = normalized * ln_weight.float()
            
            # Compare with PyTorch's built-in result
            output_pytorch = input_layernorm(embedding_output).float()
            
        results = {
            'embedding_input': x.detach().numpy(),
            'ln_weight': ln_weight.float().detach().numpy(),
            'variance': variance.detach().numpy(),
            'eps': eps,
            'variance_with_eps': variance_with_eps.detach().numpy(),
            'rms': rms.detach().numpy(),
            'normalized': normalized.detach().numpy(),
            'output_manual': output_manual.detach().numpy(),
            'output_pytorch': output_pytorch.detach().numpy(),
            'tokens': input_ids.numpy()
        }
        
        print(f"PyTorch RMS norm details:")
        print(f"  eps: {eps}")
        print(f"  variance: {variance.item():.6e}")
        print(f"  rms: {rms.item():.6e}")
        print(f"  manual vs pytorch diff: {torch.abs(output_manual - output_pytorch).max().item():.6e}")
        
        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return results, None
        
    except Exception as e:
        return None, str(e)

def manual_rms_norm_jax_detailed(x, weight, eps=1e-6):
    """Detailed manual RMS normalization for JAX with intermediate steps"""
    import jax.numpy as jnp
    
    print(f"JAX RMS norm inputs:")
    print(f"  x shape: {x.shape}")
    print(f"  weight shape: {weight.shape}")
    print(f"  eps: {eps}")
    
    # Step 1: Compute variance (mean of squares)
    variance = jnp.mean(x**2, axis=-1, keepdims=True)
    print(f"  variance: {variance.item():.6e}")
    
    # Step 2: Add epsilon
    variance_with_eps = variance + eps
    print(f"  variance_with_eps: {variance_with_eps.item():.6e}")
    
    # Step 3: Compute RMS (square root)
    rms = jnp.sqrt(variance_with_eps)
    print(f"  rms: {rms.item():.6e}")
    
    # Step 4: Normalize
    normalized = x / rms
    
    # Step 5: Scale
    output = normalized * weight
    
    return {
        'variance': variance,
        'variance_with_eps': variance_with_eps,
        'rms': rms,
        'normalized': normalized,
        'output': output
    }

def extract_jax_rms_norm_details(test_input):
    """Extract detailed RMS normalization from JAX"""
    try:
        import jax
        import jax.numpy as jnp
        sys.path.append("../..")
        
        from qwen_jax_inference import Qwen25ForCausalLM, load_params
        from transformers import AutoTokenizer
        
        print("Loading JAX model for RMS norm analysis...")
        model_path = "../../weights"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load config and model
        with open(f"{model_path}/config.json", 'r') as f:
            config = json.load(f)
        
        model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
        params = load_params(model, model_path, jnp.bfloat16)
        
        # Tokenize input
        inputs = tokenizer(test_input, return_tensors="np")
        input_ids = inputs["input_ids"]
        
        print(f"JAX input: {test_input}")
        print(f"JAX tokens: {input_ids.tolist()}")
        
        # Get embedding (confirmed identical)
        embedding_weights = params['params']['embed_tokens']['embedding']
        token_id = input_ids[0, 0]
        embedding_output = embedding_weights[token_id][None, None, :]  # [1, 1, hidden_size]
        
        # Get Layer 0 parameters
        layer_0_params = params['params']['layers_0']
        input_ln_scale = layer_0_params['input_layernorm']['scale']
        
        # Manual RMS normalization with details
        x = embedding_output.astype(jnp.float32)
        weight = input_ln_scale.astype(jnp.float32)
        
        # Try different epsilon values
        eps_values = [1e-6, 1e-5, 1e-8]
        results = {}
        
        for eps in eps_values:
            print(f"\nTrying eps = {eps}")
            rms_details = manual_rms_norm_jax_detailed(x, weight, eps)
            results[f'eps_{eps}'] = {
                'variance': np.array(rms_details['variance'], dtype=np.float32),
                'variance_with_eps': np.array(rms_details['variance_with_eps'], dtype=np.float32),
                'rms': np.array(rms_details['rms'], dtype=np.float32),
                'normalized': np.array(rms_details['normalized'], dtype=np.float32),
                'output': np.array(rms_details['output'], dtype=np.float32),
            }
        
        # Add common values
        results['embedding_input'] = np.array(x, dtype=np.float32)
        results['ln_weight'] = np.array(weight, dtype=np.float32)
        results['tokens'] = input_ids
        
        # Cleanup
        del model, params
        jax.clear_caches()
        gc.collect()
        
        return results, None
        
    except Exception as e:
        return None, str(e)

def compare_rms_norm_details(pytorch_results, jax_results):
    """Compare RMS normalization details step by step"""
    
    print("\n" + "=" * 70)
    print("RMS NORMALIZATION DETAILED ANALYSIS")
    print("=" * 70)
    
    # Compare inputs
    print(f"\n--- INPUT COMPARISON ---")
    pt_input = pytorch_results['embedding_input']
    jax_input = jax_results['embedding_input']
    
    input_diff = np.abs(pt_input - jax_input)
    print(f"Input max difference: {np.max(input_diff):.6e}")
    print(f"Input mean difference: {np.mean(input_diff):.6e}")
    
    # Compare weights
    print(f"\n--- WEIGHT COMPARISON ---")
    pt_weight = pytorch_results['ln_weight']
    jax_weight = jax_results['ln_weight']
    
    weight_diff = np.abs(pt_weight - jax_weight)
    print(f"Weight max difference: {np.max(weight_diff):.6e}")
    print(f"Weight mean difference: {np.mean(weight_diff):.6e}")
    
    # Try different epsilon values to find the match
    print(f"\n--- EPSILON COMPARISON ---")
    pt_eps = pytorch_results['eps']
    print(f"PyTorch eps: {pt_eps}")
    
    # Compare outputs for different eps values
    best_eps = None
    best_diff = float('inf')
    
    for eps_key in jax_results:
        if eps_key.startswith('eps_'):
            eps_val = float(eps_key.split('_')[1])
            jax_output = jax_results[eps_key]['output']
            pt_output = pytorch_results['output_pytorch']
            
            diff = np.abs(pt_output - jax_output)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"eps = {eps_val}: max_diff = {max_diff:.6e}, mean_diff = {mean_diff:.6e}")
            
            if max_diff < best_diff:
                best_diff = max_diff
                best_eps = eps_val
    
    print(f"\nBest eps: {best_eps} with max difference: {best_diff:.6e}")
    
    # Detailed comparison with best eps
    if best_eps:
        print(f"\n--- DETAILED COMPARISON (eps={best_eps}) ---")
        best_jax = jax_results[f'eps_{best_eps}']
        
        comparisons = [
            ('variance', pytorch_results['variance'], best_jax['variance']),
            ('rms', pytorch_results['rms'], best_jax['rms']),
            ('normalized', pytorch_results['normalized'], best_jax['normalized']),
            ('output', pytorch_results['output_pytorch'], best_jax['output'])
        ]
        
        for name, pt_data, jax_data in comparisons:
            diff = np.abs(pt_data - jax_data)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            print(f"{name:<12}: max_diff = {max_diff:.6e}, mean_diff = {mean_diff:.6e}")
    
    return best_eps, best_diff

def main():
    """Main test function"""
    print("=" * 70)
    print("PHASE 3.6: RMS NORMALIZATION DEBUG")
    print("=" * 70)
    
    test_input = get_test_input()
    print(f"Test input: '{test_input}'")
    
    # Extract PyTorch RMS norm details
    print(f"\n{'='*50}")
    print("EXTRACTING PYTORCH RMS NORM DETAILS")
    print(f"{'='*50}")
    
    pytorch_results, pytorch_error = extract_pytorch_rms_norm_details(test_input)
    if pytorch_error:
        print(f"❌ PyTorch extraction failed: {pytorch_error}")
        return False
    
    print("✅ PyTorch RMS norm details extracted")
    
    # Extract JAX RMS norm details
    print(f"\n{'='*50}")
    print("EXTRACTING JAX RMS NORM DETAILS")
    print(f"{'='*50}")
    
    jax_results, jax_error = extract_jax_rms_norm_details(test_input)
    if jax_error:
        print(f"❌ JAX extraction failed: {jax_error}")
        return False
    
    print("✅ JAX RMS norm details extracted")
    
    # Compare details
    best_eps, best_diff = compare_rms_norm_details(pytorch_results, jax_results)
    
    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 3.6 SUMMARY")
    print("=" * 70)
    
    tolerance = 1e-6
    if best_diff < tolerance:
        print(f"✅ RMS NORMALIZATION MATCHES!")
        print(f"Best eps: {best_eps}")
        print(f"Max difference: {best_diff:.6e}")
        return True
    else:
        print(f"❌ RMS NORMALIZATION STILL DIFFERS")
        print(f"Best eps: {best_eps}")
        print(f"Max difference: {best_diff:.6e}")
        print(f"Further investigation needed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 