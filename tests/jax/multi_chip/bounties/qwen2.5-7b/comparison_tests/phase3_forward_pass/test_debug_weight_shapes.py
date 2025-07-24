#!/usr/bin/env python3
"""
Debug Weight Shapes
==================
Check the exact weight shapes in PyTorch to understand the transpose issue.
"""

import sys
import os
import numpy as np

def check_pytorch_weights():
    """Check PyTorch weight shapes"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print("Loading PyTorch model to check weight shapes...")
        model_path = "../../weights"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        layer_0 = model.model.layers[0]
        attn = layer_0.self_attn
        
        print(f"PyTorch Layer 0 attention weight shapes:")
        print(f"  q_proj.weight: {attn.q_proj.weight.shape}")
        print(f"  k_proj.weight: {attn.k_proj.weight.shape}")
        print(f"  v_proj.weight: {attn.v_proj.weight.shape}")
        print(f"  o_proj.weight: {attn.o_proj.weight.shape}")
        
        print(f"\nPyTorch Layer 0 attention bias shapes:")
        print(f"  q_proj.bias: {attn.q_proj.bias.shape if attn.q_proj.bias is not None else 'None'}")
        print(f"  k_proj.bias: {attn.k_proj.bias.shape if attn.k_proj.bias is not None else 'None'}")
        print(f"  v_proj.bias: {attn.v_proj.bias.shape if attn.v_proj.bias is not None else 'None'}")
        print(f"  o_proj.bias: {attn.o_proj.bias.shape if attn.o_proj.bias is not None else 'None'}")
        
        print(f"\nConfig values:")
        print(f"  hidden_size: {model.config.hidden_size}")
        print(f"  num_attention_heads: {model.config.num_attention_heads}")
        print(f"  num_key_value_heads: {model.config.num_key_value_heads}")
        print(f"  head_dim: {model.config.hidden_size // model.config.num_attention_heads}")
        print(f"  kv_dim: {model.config.num_key_value_heads * (model.config.hidden_size // model.config.num_attention_heads)}")
        
        # Cleanup
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"Error: {e}")

def check_jax_expectations():
    """Check what JAX Dense layers expect"""
    try:
        import jax
        import jax.numpy as jnp
        sys.path.append("../..")
        
        # Check JAX model structure without loading weights
        print(f"\nJAX Dense layer expectations:")
        print(f"  For k_proj: input=3584 (hidden_size), output=512 (kv_dim)")
        print(f"  Flax Dense(512) expects kernel shape: (3584, 512) [input_features, output_features]")
        print(f"  PyTorch Linear(3584, 512) has weight shape: (512, 3584) [output_features, input_features]")
        print(f"  Therefore: PyTorch weight.T would give us the right shape for Flax")
        print(f"  Our transpose fix prevents this, so we need a different approach")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_pytorch_weights()
    check_jax_expectations() 