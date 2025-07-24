#!/usr/bin/env python3
"""
Phase 3.10: Attention Output Debug
=================================
Focused test to debug attention output computation - we know weights are identical,
so the issue is in applying weights to values or output projection.
"""

import sys
import os
import numpy as np
import time
import gc
import json
import math
from pathlib import Path

def extract_pytorch_attention_internals():
    """Extract the exact attention computation from PyTorch"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print("Loading PyTorch model for attention internals...")
        model_path = "../../weights"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model.eval()
        
        # Get single token
        inputs = tokenizer("Hello", return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            # Get normalized input (we know this is identical)
            embedding_output = model.model.embed_tokens(input_ids)
            layer_0 = model.model.layers[0]
            normalized_input = layer_0.input_layernorm(embedding_output)
            
            # Get attention details through built-in forward pass with output_attentions=True
            attn_layer = layer_0.self_attn
            attn_output, attn_weights, past_key_value = attn_layer(
                normalized_input,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=True
            )
            
            print(f"PyTorch built-in results:")
            print(f"  Attention output shape: {attn_output.shape}")
            print(f"  Attention weights shape: {attn_weights.shape}")
            print(f"  Attention output sample: {attn_output[0, 0, :5].float()}")
            print(f"  Attention weights sample: {attn_weights[0, 0, 0, 0].float()}")
            
            # Now let's do manual step-by-step to understand what's happening
            print(f"\nManual step-by-step computation:")
            
            # Q, K, V projections
            hidden_states = normalized_input
            bsz, q_len, _ = hidden_states.size()
            
            query_states = attn_layer.q_proj(hidden_states)
            key_states = attn_layer.k_proj(hidden_states)
            value_states = attn_layer.v_proj(hidden_states)
            
            print(f"  Q/K/V raw shapes: {query_states.shape}, {key_states.shape}, {value_states.shape}")
            
            # Reshape to multi-head format
            query_states = query_states.view(bsz, q_len, attn_layer.num_heads, attn_layer.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, attn_layer.num_key_value_heads, attn_layer.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, attn_layer.num_key_value_heads, attn_layer.head_dim).transpose(1, 2)
            
            print(f"  Q/K/V multi-head shapes: {query_states.shape}, {key_states.shape}, {value_states.shape}")
            
            # Apply RoPE (for single token this should be minimal)
            kv_seq_len = key_states.shape[-2]
            cos, sin = attn_layer.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb_simple(query_states, key_states, cos, sin, q_len)
            
            print(f"  Q/K after RoPE: {query_states.shape}, {key_states.shape}")
            
            # Handle GQA (repeat K/V for multiple Q heads)
            if attn_layer.num_key_value_heads != attn_layer.num_heads:
                key_states = repeat_kv_simple(key_states, attn_layer.num_heads // attn_layer.num_key_value_heads)
                value_states = repeat_kv_simple(value_states, attn_layer.num_heads // attn_layer.num_key_value_heads)
            
            print(f"  Q/K/V after GQA: {query_states.shape}, {key_states.shape}, {value_states.shape}")
            
            # Attention computation
            attn_weights_manual = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn_layer.head_dim)
            
            # For single token, no causal mask needed, but apply softmax
            # Note: PyTorch does softmax in float32 then converts back
            attn_weights_manual = torch.nn.functional.softmax(attn_weights_manual, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            print(f"  Manual attention weights shape: {attn_weights_manual.shape}")
            print(f"  Manual vs built-in weights diff: {torch.abs(attn_weights_manual - attn_weights).max().item():.6e}")
            
            # Apply attention to values
            attn_output_manual = torch.matmul(attn_weights_manual, value_states)
            
            print(f"  Attention applied to values shape: {attn_output_manual.shape}")
            
            # Reshape back
            attn_output_manual = attn_output_manual.transpose(1, 2).contiguous()
            attn_output_manual = attn_output_manual.reshape(bsz, q_len, attn_layer.hidden_size)
            
            print(f"  After reshape: {attn_output_manual.shape}")
            
            # Output projection
            attn_output_manual = attn_layer.o_proj(attn_output_manual)
            
            print(f"  After output projection: {attn_output_manual.shape}")
            print(f"  Manual vs built-in output diff: {torch.abs(attn_output_manual - attn_output).max().item():.6e}")
            
            # Check intermediate steps
            print(f"\nDetailed comparison:")
            print(f"  Built-in output sample: {attn_output[0, 0, :5].float()}")
            print(f"  Manual output sample: {attn_output_manual[0, 0, :5].float()}")
            
        # Helper functions
        def apply_rotary_pos_emb_simple(q, k, cos, sin, seq_len):
            # Simplified RoPE for single token - should be minimal impact
            # For debugging, let's just return q, k unchanged for now
            return q, k
        
        def repeat_kv_simple(hidden_states, n_rep):
            if n_rep == 1:
                return hidden_states
            batch, num_key_value_heads, slen, head_dim = hidden_states.shape
            hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
            return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        
        results = {
            'normalized_input': normalized_input.float().detach().numpy(),
            'query_states': query_states.float().detach().numpy(),
            'key_states': key_states.float().detach().numpy(), 
            'value_states': value_states.float().detach().numpy(),
            'attn_weights': attn_weights.float().detach().numpy(),
            'attn_output': attn_output.float().detach().numpy(),
            'tokens': input_ids.numpy()
        }
        
        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return results, None
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, str(e)

def main():
    """Main test function"""
    print("=" * 70)
    print("PHASE 3.10: ATTENTION OUTPUT DEBUG")
    print("=" * 70)
    
    pytorch_results, error = extract_pytorch_attention_internals()
    if error:
        print(f"❌ Failed: {error}")
        return False
    
    print("✅ PyTorch attention internals extracted")
    
    # Next: Compare with JAX implementation
    # For now, this gives us insight into PyTorch's exact computation
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 