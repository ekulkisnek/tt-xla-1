#!/usr/bin/env python3
"""
Phase 3.12: Projection Weights Debug
====================================
Debug the Q/K/V projection weights and their application since that's where divergence occurs.
"""

import sys
import os
import numpy as np
import time
import gc
import json
import math
from pathlib import Path

def compare_projection_weights():
    """Compare Q/K/V projection weights between PyTorch and JAX"""
    
    print("=" * 70)
    print("EXTRACTING PYTORCH PROJECTION WEIGHTS")
    print("=" * 70)
    
    pytorch_weights = extract_pytorch_projection_weights()
    if pytorch_weights is None:
        return False
        
    print("✅ PyTorch weights extracted")
    
    print("\n" + "=" * 70)
    print("EXTRACTING JAX PROJECTION WEIGHTS")  
    print("=" * 70)
    
    jax_weights = extract_jax_projection_weights()
    if jax_weights is None:
        return False
        
    print("✅ JAX weights extracted")
    
    print("\n" + "=" * 70)
    print("COMPARING PROJECTION WEIGHTS")
    print("=" * 70)
    
    tolerance = 1e-8
    
    for name in ['q_proj_weight', 'q_proj_bias', 'k_proj_weight', 'k_proj_bias', 'v_proj_weight', 'v_proj_bias']:
        if name not in pytorch_weights or name not in jax_weights:
            print(f"⚠️ {name} missing in one implementation")
            continue
            
        pt_weight = pytorch_weights[name]
        jax_weight = jax_weights[name]
        
        print(f"\n--- {name.upper()} ---")
        print(f"PyTorch shape: {pt_weight.shape}")
        print(f"JAX shape: {jax_weight.shape}")
        
        if pt_weight.shape != jax_weight.shape:
            print(f"❌ Shape mismatch!")
            continue
        
        diff = np.abs(pt_weight - jax_weight)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"Max difference: {max_diff:.8e}")
        print(f"Mean difference: {mean_diff:.8e}")
        
        if max_diff < tolerance:
            print(f"✅ {name} MATCHES")
        else:
            print(f"❌ {name} DIFFERS")
            print(f"Sample PyTorch: {pt_weight.flat[:5]}")
            print(f"Sample JAX: {jax_weight.flat[:5]}")
            print(f"Sample diff: {diff.flat[:5]}")
    
    # Test projection computations
    print("\n" + "=" * 70)
    print("TESTING PROJECTION COMPUTATIONS")
    print("=" * 70)
    
    test_projection_computation(pytorch_weights, jax_weights)
    
    return True

def extract_pytorch_projection_weights():
    """Extract PyTorch Q/K/V projection weights"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        model_path = "../../weights"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model.eval()
        
        # Get Layer 0 attention weights
        layer_0 = model.model.layers[0]
        attn = layer_0.self_attn
        
        weights = {
            'q_proj_weight': attn.q_proj.weight.detach().float().numpy(),
            'q_proj_bias': attn.q_proj.bias.detach().float().numpy() if attn.q_proj.bias is not None else None,
            'k_proj_weight': attn.k_proj.weight.detach().float().numpy(),
            'k_proj_bias': attn.k_proj.bias.detach().float().numpy() if attn.k_proj.bias is not None else None,
            'v_proj_weight': attn.v_proj.weight.detach().float().numpy(),
            'v_proj_bias': attn.v_proj.bias.detach().float().numpy() if attn.v_proj.bias is not None else None,
        }
        
        print(f"PyTorch projection weights:")
        for name, weight in weights.items():
            if weight is not None:
                print(f"  {name}: {weight.shape}")
            else:
                print(f"  {name}: None")
        
        # Also get normalized input for testing
        inputs = tokenizer("Hello", return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            embedding_output = model.model.embed_tokens(input_ids)
            normalized_input = layer_0.input_layernorm(embedding_output)
            weights['normalized_input'] = normalized_input.detach().float().numpy()
        
        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return weights
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

def extract_jax_projection_weights():
    """Extract JAX Q/K/V projection weights"""
    try:
        import jax
        import jax.numpy as jnp
        sys.path.append("../..")
        
        from qwen_jax_inference import Qwen25ForCausalLM, load_params
        from transformers import AutoTokenizer
        
        model_path = "../../weights"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load config and model
        with open(f"{model_path}/config.json", 'r') as f:
            config = json.load(f)
        
        model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
        params = load_params(model, model_path, jnp.bfloat16)
        
        # Get Layer 0 attention weights
        layer_0_params = params['params']['layers_0']
        
        weights = {
            'q_proj_weight': np.array(layer_0_params['self_attn']['q_proj']['kernel'].astype(jnp.float32), dtype=np.float32),
            'q_proj_bias': np.array(layer_0_params['self_attn']['q_proj']['bias'].astype(jnp.float32), dtype=np.float32),
            'k_proj_weight': np.array(layer_0_params['self_attn']['k_proj']['kernel'].astype(jnp.float32), dtype=np.float32),
            'k_proj_bias': np.array(layer_0_params['self_attn']['k_proj']['bias'].astype(jnp.float32), dtype=np.float32),
            'v_proj_weight': np.array(layer_0_params['self_attn']['v_proj']['kernel'].astype(jnp.float32), dtype=np.float32),
            'v_proj_bias': np.array(layer_0_params['self_attn']['v_proj']['bias'].astype(jnp.float32), dtype=np.float32),
        }
        
        print(f"JAX projection weights:")
        for name, weight in weights.items():
            if weight is not None:
                print(f"  {name}: {weight.shape}")
            else:
                print(f"  {name}: None")
        
        # Also get normalized input for testing
        inputs = tokenizer("Hello", return_tensors="np")
        input_ids = inputs["input_ids"]
        
        embedding_weights = params['params']['embed_tokens']['embedding']
        token_id = input_ids[0, 0]
        embedding_output = embedding_weights[token_id][None, None, :]
        
        # Manual RMS norm
        input_ln_scale = layer_0_params['input_layernorm']['scale']
        
        def manual_rms_norm_jax(x, weight, eps=1e-6):
            input_dtype = x.dtype
            hidden_states = x.astype(jnp.float32)
            variance = jnp.mean(hidden_states**2, axis=-1, keepdims=True)
            hidden_states = hidden_states * jnp.power(variance + eps, -0.5)
            hidden_states = hidden_states.astype(input_dtype)
            return weight * hidden_states
        
        normalized_input = manual_rms_norm_jax(embedding_output, input_ln_scale)
        weights['normalized_input'] = np.array(normalized_input.astype(jnp.float32), dtype=np.float32)
        
        # Cleanup
        del model, params
        jax.clear_caches()
        gc.collect()
        
        return weights
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

def test_projection_computation(pytorch_weights, jax_weights):
    """Test the actual projection computation to see where divergence occurs"""
    
    print("Testing projection computation...")
    
    # Get normalized inputs (should be identical)
    pt_input = pytorch_weights['normalized_input']
    jax_input = jax_weights['normalized_input']
    
    input_diff = np.max(np.abs(pt_input - jax_input))
    print(f"Normalized input difference: {input_diff:.8e}")
    
    if input_diff > 1e-6:
        print("❌ Normalized inputs differ - this shouldn't happen!")
        return
    
    # Test each projection
    for proj in ['q', 'k', 'v']:
        print(f"\nTesting {proj}_proj computation:")
        
        pt_weight = pytorch_weights[f'{proj}_proj_weight']
        pt_bias = pytorch_weights[f'{proj}_proj_bias']
        jax_weight = jax_weights[f'{proj}_proj_weight']
        jax_bias = jax_weights[f'{proj}_proj_bias']
        
        # PyTorch computation: output = input @ weight.T + bias
        pt_output = np.dot(pt_input, pt_weight.T) + pt_bias
        
        # JAX computation: output = input @ weight + bias
        jax_output = np.dot(jax_input, jax_weight) + jax_bias
        
        output_diff = np.max(np.abs(pt_output - jax_output))
        print(f"  {proj}_proj output difference: {output_diff:.8e}")
        
        if output_diff > 1e-6:
            print(f"  ❌ {proj}_proj outputs differ")
            
            # Check if it's the transpose issue
            jax_output_transposed = np.dot(jax_input, jax_weight.T) + jax_bias
            transpose_diff = np.max(np.abs(pt_output - jax_output_transposed))
            print(f"  With transpose, difference: {transpose_diff:.8e}")
            
            if transpose_diff < 1e-6:
                print(f"  🎯 FOUND ISSUE: {proj}_proj needs weight transpose!")
            else:
                print(f"  🤔 Issue is not simple transpose")
        else:
            print(f"  ✅ {proj}_proj outputs match")

def main():
    """Main test function"""
    print("=" * 70)
    print("PHASE 3.12: PROJECTION WEIGHTS DEBUG")
    print("=" * 70)
    
    success = compare_projection_weights()
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 