#!/usr/bin/env python3
"""
Phase 3B.1: Attention Mechanism Deep Dive
==========================================
Deep analysis of attention computation differences between PyTorch and JAX.
We know from Phase 3A.1 that attention output differs by 2.44, causing all downstream issues.
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
    """Get test input for attention analysis"""
    return "Hello"  # Token 9707 - confirmed identical embedding

def extract_pytorch_attention_details(test_input):
    """Extract detailed attention computation from PyTorch"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print("Loading PyTorch model for attention analysis...")
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
        
        # Extract step-by-step attention computation
        with torch.no_grad():
            # Get to attention input
            embedding_output = model.model.embed_tokens(input_ids)
            layer_0 = model.model.layers[0]
            normalized_input = layer_0.input_layernorm(embedding_output)
            
            # Manual attention computation to extract intermediates
            attn = layer_0.self_attn
            hidden_states = normalized_input
            bsz, q_len, _ = hidden_states.size()
            
            # Q, K, V projections  
            query_states = attn.q_proj(hidden_states)
            key_states = attn.k_proj(hidden_states)  
            value_states = attn.v_proj(hidden_states)
            
            print(f"PyTorch Q/K/V after projection:")
            print(f"  query_states: {query_states.shape}")
            print(f"  key_states: {key_states.shape}")
            print(f"  value_states: {value_states.shape}")
            
            # Reshape for multi-head attention
            query_states = query_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            
            print(f"PyTorch Q/K/V after reshape:")
            print(f"  query_states: {query_states.shape}")
            print(f"  key_states: {key_states.shape}")
            print(f"  value_states: {value_states.shape}")
            
            # Apply RoPE - this is crucial!
            position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0)
            
            # SKIP RoPE for now to isolate other differences
            print("Warning: Skipping RoPE in PyTorch for initial analysis")
            query_states_after_rope = query_states  # No RoPE applied
            key_states_after_rope = key_states      # No RoPE applied
            
            print(f"PyTorch Q/K after RoPE:")
            print(f"  query_states: {query_states_after_rope.shape}")
            print(f"  key_states: {key_states_after_rope.shape}")
            
            # Apply GQA - expand key/value heads
            if attn.num_key_value_heads != attn.num_heads:
                key_states_after_rope = key_states_after_rope.repeat_interleave(attn.num_heads // attn.num_key_value_heads, dim=1)
                value_states = value_states.repeat_interleave(attn.num_heads // attn.num_key_value_heads, dim=1)
            
            print(f"PyTorch K/V after GQA expansion:")
            print(f"  key_states: {key_states_after_rope.shape}")
            print(f"  value_states: {value_states.shape}")
            
            # Attention computation
            attn_weights = torch.matmul(query_states_after_rope, key_states_after_rope.transpose(2, 3)) / math.sqrt(attn.head_dim)
            
            print(f"PyTorch attention weights: {attn_weights.shape}")
            
            # Softmax
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            # Apply attention to values
            attn_output = torch.matmul(attn_weights, value_states)
            
            print(f"PyTorch attention output before projection: {attn_output.shape}")
            
            # Reshape back
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, attn.hidden_size)
            
            # Output projection
            final_attn_output = attn.o_proj(attn_output)
            
            print(f"PyTorch final attention output: {final_attn_output.shape}")
            
        # Store all intermediates  
        results = {
            'normalized_input': normalized_input.detach().float().numpy(),
            'query_raw': attn.q_proj(hidden_states).detach().float().numpy(),
            'key_raw': attn.k_proj(hidden_states).detach().float().numpy(),
            'value_raw': attn.v_proj(hidden_states).detach().float().numpy(),
            'query_reshaped': query_states_after_rope.detach().float().numpy(),
            'key_reshaped': key_states_after_rope.detach().float().numpy(), 
            'value_reshaped': value_states.detach().float().numpy(),
            'attn_weights': attn_weights.detach().float().numpy(),
            'attn_output_pre_proj': attn_output.detach().float().numpy(),
            'final_attn_output': final_attn_output.detach().float().numpy(),
            'tokens': input_ids.numpy()
        }
        
        print(f"PyTorch attention intermediates extracted")
        
        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return results, None
        
    except Exception as e:
        import traceback
        return None, f"{str(e)}\n{traceback.format_exc()}"

def extract_jax_attention_details(test_input):
    """Extract detailed attention computation from JAX"""
    try:
        import jax
        import jax.numpy as jnp
        sys.path.append("../..")
        
        from qwen_jax_inference import Qwen25ForCausalLM, load_params
        from transformers import AutoTokenizer
        
        print("Loading JAX model for attention analysis...")
        model_path = "../../weights"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load config and model
        with open(f"{model_path}/config.json", 'r') as f:
            config = json.load(f)
        
        model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
        params = load_params(model, model_path, jnp.bfloat16)
        
        # Tokenize input
        inputs = tokenizer(test_input, return_tensors="np")
        input_ids = jnp.array(inputs["input_ids"])
        
        print(f"JAX input: {test_input}")
        print(f"JAX tokens: {input_ids.tolist()}")
        
        # Get inputs to attention
        batch_size, seq_len = input_ids.shape
        
        # Embedding and normalization
        embedding_output = model.apply(
            {'params': params['params']}, 
            input_ids,
            method=lambda module, input_ids: module.embed_tokens(input_ids)
        )
        
        normalized_input = model.apply(
            {'params': params['params']}, 
            embedding_output,
            method=lambda module, x: module.layers[0].input_layernorm(x)
        )
        
        # Manual attention computation to extract intermediates
        layer_0_params = params['params']['layers_0']['self_attn']
        
        # Q, K, V projections using weights directly
        q_weights = layer_0_params['q_proj']['kernel']
        q_bias = layer_0_params['q_proj']['bias']
        k_weights = layer_0_params['k_proj']['kernel'] 
        k_bias = layer_0_params['k_proj']['bias']
        v_weights = layer_0_params['v_proj']['kernel']
        v_bias = layer_0_params['v_proj']['bias']
        
        print(f"JAX projection weight shapes:")
        print(f"  q_weights: {q_weights.shape}")
        print(f"  k_weights: {k_weights.shape}")
        print(f"  v_weights: {v_weights.shape}")
        
        # Apply projections
        query_raw = jnp.dot(normalized_input, q_weights) + q_bias
        key_raw = jnp.dot(normalized_input, k_weights) + k_bias
        value_raw = jnp.dot(normalized_input, v_weights) + v_bias
        
        print(f"JAX Q/K/V after projection:")
        print(f"  query_raw: {query_raw.shape}")
        print(f"  key_raw: {key_raw.shape}")
        print(f"  value_raw: {value_raw.shape}")
        
        # Reshape for attention
        num_heads = config['num_attention_heads']
        num_kv_heads = config['num_key_value_heads']
        head_dim = config['hidden_size'] // num_heads
        
        query_reshaped = query_raw.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        key_reshaped = key_raw.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        value_reshaped = value_raw.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        
        print(f"JAX Q/K/V after reshape:")
        print(f"  query_reshaped: {query_reshaped.shape}")
        print(f"  key_reshaped: {key_reshaped.shape}")
        print(f"  value_reshaped: {value_reshaped.shape}")
        
        # Apply RoPE (using the manual implementation from qwen_jax_inference.py)
        position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
        
        # SKIP RoPE for now to isolate other differences
        print("Warning: Skipping RoPE in JAX for initial analysis")
        query_with_rope = query_reshaped  # No RoPE applied
        key_with_rope = key_reshaped      # No RoPE applied
        
        print(f"JAX Q/K after RoPE:")
        print(f"  query_with_rope: {query_with_rope.shape}")
        print(f"  key_with_rope: {key_with_rope.shape}")
        
        # Apply GQA - expand key/value heads
        if num_kv_heads != num_heads:
            repeat = num_heads // num_kv_heads
            key_expanded = jnp.repeat(key_with_rope, repeat, axis=1)
            value_expanded = jnp.repeat(value_reshaped, repeat, axis=1)
        else:
            key_expanded = key_with_rope
            value_expanded = value_reshaped
        
        print(f"JAX K/V after GQA expansion:")
        print(f"  key_expanded: {key_expanded.shape}")
        print(f"  value_expanded: {value_expanded.shape}")
        
        # Attention computation
        scale = 1.0 / jnp.sqrt(head_dim)
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', query_with_rope, key_expanded) * scale
        
        print(f"JAX attention weights: {attn_weights.shape}")
        
        # Softmax
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        
        # Apply attention to values
        attn_output_pre_proj = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, value_expanded)
        
        print(f"JAX attention output before projection: {attn_output_pre_proj.shape}")
        
        # Reshape back
        attn_output_reshaped = attn_output_pre_proj.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, config['hidden_size'])
        
        # Output projection
        o_weights = layer_0_params['o_proj']['kernel']
        final_attn_output = jnp.dot(attn_output_reshaped, o_weights)
        
        print(f"JAX final attention output: {final_attn_output.shape}")
        
        # Store all intermediates
        results = {
            'normalized_input': np.array(normalized_input.astype(jnp.float32)),
            'query_raw': np.array(query_raw.astype(jnp.float32)),
            'key_raw': np.array(key_raw.astype(jnp.float32)),
            'value_raw': np.array(value_raw.astype(jnp.float32)),
            'query_reshaped': np.array(query_reshaped.astype(jnp.float32)),
            'key_reshaped': np.array(key_reshaped.astype(jnp.float32)),
            'value_reshaped': np.array(value_reshaped.astype(jnp.float32)),
            'attn_weights': np.array(attn_weights.astype(jnp.float32)),
            'attn_output_pre_proj': np.array(attn_output_reshaped.astype(jnp.float32)),
            'final_attn_output': np.array(final_attn_output.astype(jnp.float32)),
            'tokens': np.array(input_ids)
        }
        
        print(f"JAX attention intermediates extracted")
        
        # Cleanup
        del model, params
        jax.clear_caches()
        gc.collect()
        
        return results, None
        
    except Exception as e:
        import traceback
        return None, f"{str(e)}\n{traceback.format_exc()}"

def compare_attention_details(pytorch_results, jax_results):
    """Compare attention computation step by step"""
    print("\n" + "=" * 70)
    print("DETAILED ATTENTION MECHANISM COMPARISON")
    print("=" * 70)
    
    tolerance = 1e-6
    medium_tolerance = 1e-4
    large_tolerance = 1e-2
    
    components = [
        'normalized_input',
        'query_raw', 
        'key_raw',
        'value_raw',
        'query_reshaped',
        'key_reshaped',
        'value_reshaped',
        'attn_weights',
        'attn_output_pre_proj',
        'final_attn_output'
    ]
    
    results_summary = {}
    first_major_difference = None
    
    for component in components:
        if component not in pytorch_results or component not in jax_results:
            print(f"⚠️ {component} missing in one implementation")
            continue
            
        pt_data = pytorch_results[component]
        jax_data = jax_results[component]
        
        print(f"\n--- {component.upper()} ---")
        print(f"PyTorch shape: {pt_data.shape}")
        print(f"JAX shape: {jax_data.shape}")
        
        if pt_data.shape != jax_data.shape:
            print(f"❌ Shape mismatch!")
            continue
        
        diff = np.abs(pt_data - jax_data)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"Max difference: {max_diff:.8e}")
        print(f"Mean difference: {mean_diff:.8e}")
        
        # Categorize the difference
        if max_diff < tolerance:
            print(f"✅ {component} PERFECT MATCH (< 1e-6)")
            status = "PERFECT"
        elif max_diff < medium_tolerance:
            print(f"🟡 {component} CLOSE (< 1e-4)")
            status = "CLOSE"
        elif max_diff < large_tolerance:
            print(f"🟠 {component} MODERATE DIFFERENCE (< 1e-2)")
            status = "MODERATE"
            if first_major_difference is None:
                first_major_difference = component
        else:
            print(f"❌ {component} LARGE DIFFERENCE (> 1e-2)")
            status = "LARGE"
            if first_major_difference is None:
                first_major_difference = component
            
            # Show sample differences for large differences
            print(f"Sample PyTorch: {pt_data.flat[:5]}")
            print(f"Sample JAX: {jax_data.flat[:5]}")
            print(f"Sample diff: {diff.flat[:5]}")
        
        results_summary[component] = {
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'status': status
        }
    
    # Summary
    print("\n" + "=" * 70)
    print("ATTENTION MECHANISM ANALYSIS SUMMARY")
    print("=" * 70)
    
    perfect_count = sum(1 for r in results_summary.values() if r['status'] == 'PERFECT')
    close_count = sum(1 for r in results_summary.values() if r['status'] == 'CLOSE')
    moderate_count = sum(1 for r in results_summary.values() if r['status'] == 'MODERATE')
    large_count = sum(1 for r in results_summary.values() if r['status'] == 'LARGE')
    
    print(f"Perfect matches (< 1e-6): {perfect_count}/{len(results_summary)}")
    print(f"Close matches (< 1e-4): {close_count}/{len(results_summary)}")
    print(f"Moderate differences (< 1e-2): {moderate_count}/{len(results_summary)}")
    print(f"Large differences (> 1e-2): {large_count}/{len(results_summary)}")
    
    if first_major_difference:
        print(f"\n🎯 FIRST MAJOR DIFFERENCE: {first_major_difference}")
        print(f"This is where the attention computation diverges significantly.")
    
    success = perfect_count == len(results_summary)
    
    return success, results_summary, first_major_difference

def main():
    """Run Phase 3B.1: Attention Mechanism Deep Dive"""
    print("=" * 70)
    print("PHASE 3B.1: ATTENTION MECHANISM DEEP DIVE")
    print("=" * 70)
    print("Analyzing attention computation step-by-step to identify divergence point")
    
    test_input = get_test_input()
    
    # Extract PyTorch attention details
    print(f"\n🔄 Extracting PyTorch attention details...")
    pytorch_results, pt_error = extract_pytorch_attention_details(test_input)
    if pytorch_results is None:
        print(f"❌ PyTorch extraction failed: {pt_error}")
        return False
    
    # Extract JAX attention details  
    print(f"\n🔄 Extracting JAX attention details...")
    jax_results, jax_error = extract_jax_attention_details(test_input)
    if jax_results is None:
        print(f"❌ JAX extraction failed: {jax_error}")
        return False
    
    # Compare step by step
    print(f"\n🔄 Comparing attention computation step by step...")
    success, summary, first_diff = compare_attention_details(pytorch_results, jax_results)
    
    if success:
        print("\n🎉 PHASE 3B.1 COMPLETE: All attention components match perfectly!")
        print("✅ Attention mechanism implementation is identical")
    else:
        print(f"\n🔍 PHASE 3B.1 ANALYSIS COMPLETE: Issue isolated to {first_diff}")
        print("📋 Next: Investigate specific component implementation differences")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 