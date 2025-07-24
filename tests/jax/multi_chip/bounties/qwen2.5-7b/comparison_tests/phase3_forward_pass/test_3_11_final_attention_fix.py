#!/usr/bin/env python3
"""
Phase 3.11: Final Attention Fix
==============================
Focused test to debug and fix the final attention matrix multiplication issue.
We know attention weights are identical (0.00e+00) but output differs by 1.56e-02.
"""

import sys
import os
import numpy as np
import time
import gc
import json
import math
from pathlib import Path

def extract_attention_tensors():
    """Extract exact attention tensors from both PyTorch and JAX for comparison"""
    
    # Get PyTorch tensors
    print("=" * 50)
    print("EXTRACTING PYTORCH ATTENTION TENSORS")
    print("=" * 50)
    
    pytorch_tensors = extract_pytorch_tensors()
    if pytorch_tensors is None:
        return None, None
    
    print("✅ PyTorch tensors extracted")
    
    # Get JAX tensors  
    print("\n" + "=" * 50)
    print("EXTRACTING JAX ATTENTION TENSORS")
    print("=" * 50)
    
    jax_tensors = extract_jax_tensors()
    if jax_tensors is None:
        return None, None
        
    print("✅ JAX tensors extracted")
    
    return pytorch_tensors, jax_tensors

def extract_pytorch_tensors():
    """Extract PyTorch attention computation tensors"""
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
        
        inputs = tokenizer("Hello", return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            # Get to attention computation point
            embedding_output = model.model.embed_tokens(input_ids)
            layer_0 = model.model.layers[0]
            normalized_input = layer_0.input_layernorm(embedding_output)
            
            # Manual attention step-by-step
            attn = layer_0.self_attn
            hidden_states = normalized_input
            bsz, q_len, _ = hidden_states.size()
            
            # Q, K, V projections
            query_states = attn.q_proj(hidden_states)
            key_states = attn.k_proj(hidden_states)  
            value_states = attn.v_proj(hidden_states)
            
            # Reshape
            query_states = query_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            
            # Skip RoPE for debugging (we know it's not the issue)
            # Apply GQA
            if attn.num_key_value_heads != attn.num_heads:
                key_states = key_states.repeat_interleave(attn.num_heads // attn.num_key_value_heads, dim=1)
                value_states = value_states.repeat_interleave(attn.num_heads // attn.num_key_value_heads, dim=1)
            
            # Attention computation
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn.head_dim)
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            # Apply attention to values - THIS IS WHERE THE ISSUE IS
            attn_output = torch.matmul(attn_weights, value_states)
            
            # Reshape back
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, attn.hidden_size)
            
            # Output projection
            final_output = attn.o_proj(attn_output)
            
        tensors = {
            'value_states': value_states.float().detach().numpy(),
            'attn_weights': attn_weights.float().detach().numpy(),
            'attn_output_before_proj': attn_output.float().detach().numpy(),
            'final_output': final_output.float().detach().numpy(),
            'query_states': query_states.float().detach().numpy(),
            'key_states': key_states.float().detach().numpy(),
        }
        
        print(f"PyTorch tensor shapes:")
        for name, tensor in tensors.items():
            print(f"  {name}: {tensor.shape}")
        
        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return tensors
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

def extract_jax_tensors():
    """Extract JAX attention computation tensors"""
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
        
        inputs = tokenizer("Hello", return_tensors="np")
        input_ids = inputs["input_ids"]
        
        # Get to attention computation (replicating our fixed implementation)
        embedding_weights = params['params']['embed_tokens']['embedding']
        token_id = input_ids[0, 0]
        embedding_output = embedding_weights[token_id][None, None, :]
        
        # RMS norm (we know this is perfect)
        layer_0_params = params['params']['layers_0']
        input_ln_scale = layer_0_params['input_layernorm']['scale']
        
        # Manual RMS norm (our fixed version)
        def manual_rms_norm_jax(x, weight, eps=1e-6):
            input_dtype = x.dtype
            hidden_states = x.astype(jnp.float32)
            variance = jnp.mean(hidden_states**2, axis=-1, keepdims=True)
            hidden_states = hidden_states * jnp.power(variance + eps, -0.5)
            hidden_states = hidden_states.astype(input_dtype)
            return weight * hidden_states
        
        normalized_input = manual_rms_norm_jax(embedding_output, input_ln_scale)
        
        # Attention computation (matching our implementation)
        attn_q_kernel = layer_0_params['self_attn']['q_proj']['kernel']
        attn_q_bias = layer_0_params['self_attn']['q_proj']['bias']
        attn_k_kernel = layer_0_params['self_attn']['k_proj']['kernel']
        attn_k_bias = layer_0_params['self_attn']['k_proj']['bias']
        attn_v_kernel = layer_0_params['self_attn']['v_proj']['kernel']
        attn_v_bias = layer_0_params['self_attn']['v_proj']['bias']
        attn_o_kernel = layer_0_params['self_attn']['o_proj']['kernel']
        attn_o_bias = layer_0_params['self_attn']['o_proj'].get('bias', None)
        
        batch_size, seq_len, hidden_size = normalized_input.shape
        num_heads = config['num_attention_heads']
        num_key_value_heads = config.get('num_key_value_heads', num_heads)
        head_dim = hidden_size // num_heads
        
        # Q, K, V projections
        q = jnp.dot(normalized_input, attn_q_kernel.astype(jnp.bfloat16)) + attn_q_bias.astype(jnp.bfloat16)
        k = jnp.dot(normalized_input, attn_k_kernel.astype(jnp.bfloat16)) + attn_k_bias.astype(jnp.bfloat16)
        v = jnp.dot(normalized_input, attn_v_kernel.astype(jnp.bfloat16)) + attn_v_bias.astype(jnp.bfloat16)
        
        # Reshape
        q = q.reshape(batch_size, seq_len, num_heads, head_dim)
        k = k.reshape(batch_size, seq_len, num_key_value_heads, head_dim)
        v = v.reshape(batch_size, seq_len, num_key_value_heads, head_dim)
        
        # Skip RoPE for debugging (q_rope, k_rope = q, k)
        
        # Transpose for attention
        q_t = jnp.transpose(q, (0, 2, 1, 3))
        k_t = jnp.transpose(k, (0, 2, 1, 3))
        v_t = jnp.transpose(v, (0, 2, 1, 3))
        
        # GQA expansion
        if num_key_value_heads != num_heads:
            heads_per_group = num_heads // num_key_value_heads
            k_t = jnp.repeat(k_t, heads_per_group, axis=1)
            v_t = jnp.repeat(v_t, heads_per_group, axis=1)
        
        # Attention computation (mixed precision)
        q_t_f32 = q_t.astype(jnp.float32)
        k_t_f32 = k_t.astype(jnp.float32)
        v_t_f32 = v_t.astype(jnp.float32)
        
        attn_scores = jnp.matmul(q_t_f32, jnp.transpose(k_t_f32, (0, 1, 3, 2))) / math.sqrt(head_dim)
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        
        # Apply attention - THIS IS THE CRITICAL STEP
        attn_output = jnp.matmul(attn_weights, v_t_f32)
        attn_output = attn_output.astype(q.dtype)  # Convert back to bfloat16
        
        # Reshape back
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)
        
        # Output projection
        final_output = jnp.dot(attn_output, attn_o_kernel.astype(jnp.bfloat16))
        if attn_o_bias is not None:
            final_output = final_output + attn_o_bias.astype(jnp.bfloat16)
        
        tensors = {
            'value_states': np.array(v_t.astype(jnp.float32), dtype=np.float32),
            'attn_weights': np.array(attn_weights.astype(jnp.float32), dtype=np.float32),
            'attn_output_before_proj': np.array(attn_output.astype(jnp.float32), dtype=np.float32),
            'final_output': np.array(final_output.astype(jnp.float32), dtype=np.float32),
            'query_states': np.array(q_t.astype(jnp.float32), dtype=np.float32),
            'key_states': np.array(k_t.astype(jnp.float32), dtype=np.float32),
        }
        
        print(f"JAX tensor shapes:")
        for name, tensor in tensors.items():
            print(f"  {name}: {tensor.shape}")
        
        # Cleanup
        del model, params
        jax.clear_caches()
        gc.collect()
        
        return tensors
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

def compare_attention_tensors(pytorch_tensors, jax_tensors):
    """Compare attention tensors step by step to find the exact divergence"""
    
    print("\n" + "=" * 70)
    print("STEP-BY-STEP ATTENTION TENSOR COMPARISON")
    print("=" * 70)
    
    tolerance = 1e-6
    
    for name in ['query_states', 'key_states', 'value_states', 'attn_weights', 'attn_output_before_proj', 'final_output']:
        if name not in pytorch_tensors or name not in jax_tensors:
            continue
            
        pt_tensor = pytorch_tensors[name]
        jax_tensor = jax_tensors[name]
        
        print(f"\n--- {name.upper()} ---")
        print(f"PyTorch shape: {pt_tensor.shape}")
        print(f"JAX shape: {jax_tensor.shape}")
        
        if pt_tensor.shape != jax_tensor.shape:
            print(f"❌ Shape mismatch!")
            continue
        
        diff = np.abs(pt_tensor - jax_tensor)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"Max difference: {max_diff:.6e}")
        print(f"Mean difference: {mean_diff:.6e}")
        print(f"Tolerance: {tolerance:.6e}")
        
        if max_diff < tolerance:
            print(f"✅ {name} MATCHES")
        else:
            print(f"❌ {name} DIFFERS")
            
            # Show some sample values for debugging
            print(f"PyTorch sample: {pt_tensor.flat[:5]}")
            print(f"JAX sample: {jax_tensor.flat[:5]}")
            print(f"Diff sample: {diff.flat[:5]}")
            
            # If this is the attention output, do deeper debugging
            if name == 'attn_output_before_proj':
                print(f"\n🎯 CRITICAL: This is where divergence occurs!")
                print(f"Investigating matrix multiplication...")
                
                # Check if it's the matmul itself
                pt_weights = pytorch_tensors['attn_weights']
                pt_values = pytorch_tensors['value_states']
                jax_weights = jax_tensors['attn_weights']
                jax_values = jax_tensors['value_states']
                
                print(f"Weights match: {np.allclose(pt_weights, jax_weights, atol=tolerance)}")
                print(f"Values match: {np.allclose(pt_values, jax_values, atol=tolerance)}")
                
                if np.allclose(pt_weights, jax_weights, atol=tolerance) and np.allclose(pt_values, jax_values, atol=tolerance):
                    print(f"🚨 CRITICAL: Inputs identical but outputs differ!")
                    print(f"This suggests a fundamental difference in matrix multiplication")
                    
                    # Test manual matrix multiplication
                    manual_pt = np.matmul(pt_weights, pt_values)
                    manual_jax = np.matmul(jax_weights, jax_values)
                    manual_diff = np.max(np.abs(manual_pt - manual_jax))
                    
                    print(f"Manual matmul difference: {manual_diff:.6e}")
                    
                    if manual_diff < tolerance:
                        print(f"✅ Manual matmul identical - issue in JAX implementation")
                        return "implementation_issue"
                    else:
                        print(f"❌ Manual matmul differs - deeper numerical issue")
                        return "numerical_issue"
    
    return "unknown"

def main():
    """Main test function"""
    print("=" * 70)
    print("PHASE 3.11: FINAL ATTENTION FIX")
    print("=" * 70)
    
    pytorch_tensors, jax_tensors = extract_attention_tensors()
    if pytorch_tensors is None or jax_tensors is None:
        print("❌ Failed to extract tensors")
        return False
    
    issue_type = compare_attention_tensors(pytorch_tensors, jax_tensors)
    
    print(f"\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)
    print(f"Issue type: {issue_type}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 