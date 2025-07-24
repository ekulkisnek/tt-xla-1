#!/usr/bin/env python3
"""
Phase 3.9: Attention Mechanism Debug
===================================
Focused test to debug attention computation differences between PyTorch and JAX.
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
    return "Hello"  # Token 9707 - confirmed identical embedding and normalization

def extract_pytorch_attention_details(test_input):
    """Extract detailed attention components from PyTorch"""
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
        
        with torch.no_grad():
            # Get the normalized input (we know this matches now)
            embedding_output = model.model.embed_tokens(input_ids)
            layer_0 = model.model.layers[0]
            normalized_input = layer_0.input_layernorm(embedding_output)
            
            # Get attention layer
            attention = layer_0.self_attn
            
            print(f"Attention layer type: {type(attention)}")
            
            # Manual attention computation step by step
            hidden_states = normalized_input
            bsz, q_len, _ = hidden_states.size()
            
            # Q, K, V projections
            query_states = attention.q_proj(hidden_states)
            key_states = attention.k_proj(hidden_states)
            value_states = attention.v_proj(hidden_states)
            
            print(f"Q/K/V shapes after projection: {query_states.shape}, {key_states.shape}, {value_states.shape}")
            
            # Reshape for multi-head attention
            query_states = query_states.view(bsz, q_len, attention.num_heads, attention.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
            
            print(f"Q/K/V shapes after reshape: {query_states.shape}, {key_states.shape}, {value_states.shape}")
            
            # Apply rotary position embedding (RoPE)
            cos, sin = attention.rotary_emb(value_states, seq_len=q_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, None)
            
            print(f"Q/K shapes after RoPE: {query_states.shape}, {key_states.shape}")
            
            # Repeat key and value for grouped query attention if needed
            if attention.num_key_value_heads != attention.num_heads:
                key_states = repeat_kv(key_states, attention.num_heads // attention.num_key_value_heads)
                value_states = repeat_kv(value_states, attention.num_heads // attention.num_key_value_heads)
                
            print(f"Q/K/V shapes after GQA: {query_states.shape}, {key_states.shape}, {value_states.shape}")
            
            # Attention computation
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attention.head_dim)
            
            print(f"Attention weights shape: {attn_weights.shape}")
            
            # Apply causal mask (not needed for single token but let's be explicit)
            if q_len > 1:
                mask = torch.triu(torch.ones(q_len, q_len, dtype=torch.bool), diagonal=1)
                attn_weights.masked_fill_(mask.unsqueeze(0).unsqueeze(0), torch.finfo(attn_weights.dtype).min)
            
            # Softmax
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            # Apply attention to values
            attn_output = torch.matmul(attn_weights, value_states)
            
            print(f"Attention output shape: {attn_output.shape}")
            
            # Transpose and reshape
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, attention.hidden_size)
            
            print(f"Attention output after reshape: {attn_output.shape}")
            
            # Output projection
            attn_output = attention.o_proj(attn_output)
            
            print(f"Final attention output: {attn_output.shape}")
            
        # Helper functions (from transformers library)
        def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
            # RoPE implementation - this is a simplified version
            # In practice, this is more complex
            cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
            sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
            cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
            sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)
            return q_embed, k_embed
        
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
        
        def repeat_kv(hidden_states, n_rep):
            batch, num_key_value_heads, slen, head_dim = hidden_states.shape
            if n_rep == 1:
                return hidden_states
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
        
        print(f"PyTorch attention details extracted")
        
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
    print("PHASE 3.9: ATTENTION MECHANISM DEBUG")
    print("=" * 70)
    
    test_input = get_test_input()
    print(f"Test input: '{test_input}'")
    
    # Extract PyTorch attention details
    print(f"\n{'='*50}")
    print("EXTRACTING PYTORCH ATTENTION DETAILS")
    print(f"{'='*50}")
    
    pytorch_results, pytorch_error = extract_pytorch_attention_details(test_input)
    if pytorch_error:
        print(f"❌ PyTorch extraction failed: {pytorch_error}")
        return False
    
    print("✅ PyTorch attention details extracted")
    
    # TODO: Extract JAX attention details and compare
    # This will be implemented next
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 