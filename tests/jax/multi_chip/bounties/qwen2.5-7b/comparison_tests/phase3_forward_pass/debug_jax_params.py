#!/usr/bin/env python3
"""
Debug JAX Parameter Structure
============================
Simple script to inspect the actual parameter names and structure.
"""

import sys
import os
import json
sys.path.append("../..")

def inspect_jax_params():
    """Inspect JAX model parameter structure"""
    try:
        import jax
        import jax.numpy as jnp
        
        # Import JAX model components
        from qwen_jax_inference import Qwen25ForCausalLM, load_params
        from transformers import AutoTokenizer
        
        print("Loading JAX model for parameter inspection...")
        model_path = "../../weights"
        
        # Load config
        with open(f"{model_path}/config.json", 'r') as f:
            config = json.load(f)
        
        # Create model
        model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
        
        # Load weights
        params = load_params(model, model_path, jnp.bfloat16)
        
        print("Parameter structure:")
        print(f"Top level keys: {list(params.keys())}")
        
        if 'params' in params:
            print(f"params keys: {list(params['params'].keys())}")
            
            # Check embedding structure
            if 'embed_tokens' in params['params']:
                print(f"embed_tokens keys: {list(params['params']['embed_tokens'].keys())}")
                embed_weights = params['params']['embed_tokens']
                for key, value in embed_weights.items():
                    print(f"  {key}: shape {value.shape}")
            
            # Check layer 0 structure in detail
            if 'layers_0' in params['params']:
                layer_0_params = params['params']['layers_0']
                print(f"\nDetailed Layer 0 structure:")
                print(f"layers_0 keys: {list(layer_0_params.keys())}")
                
                # Input layernorm
                if 'input_layernorm' in layer_0_params:
                    input_ln = layer_0_params['input_layernorm']
                    print("Input layernorm type:", type(input_ln))
                    print("Input layernorm keys:", list(input_ln.keys()) if hasattr(input_ln, 'keys') else 'Not a dict')
                    if hasattr(input_ln, 'keys'):
                        for key, value in input_ln.items():
                            print(f"  input_layernorm.{key}: shape {value.shape}")
                
                # Self attention
                if 'self_attn' in layer_0_params:
                    self_attn = layer_0_params['self_attn']
                    print(f"\nself_attn keys: {list(self_attn.keys())}")
                    
                    # Check all attention projections
                    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                        if proj_name in self_attn:
                            proj = self_attn[proj_name]
                            print(f"{proj_name} type:", type(proj))
                            print(f"{proj_name} keys:", list(proj.keys()) if hasattr(proj, 'keys') else 'Not a dict')
                            if hasattr(proj, 'keys'):
                                for key, value in proj.items():
                                    print(f"  {proj_name}.{key}: shape {value.shape}")
                
                # Post attention layernorm  
                if 'post_attention_layernorm' in layer_0_params:
                    post_ln = layer_0_params['post_attention_layernorm']
                    print("Post layernorm type:", type(post_ln))
                    print("Post layernorm keys:", list(post_ln.keys()) if hasattr(post_ln, 'keys') else 'Not a dict')
                    if hasattr(post_ln, 'keys'):
                        for key, value in post_ln.items():
                            print(f"  post_attention_layernorm.{key}: shape {value.shape}")
                
                # MLP
                if 'mlp' in layer_0_params:
                    mlp = layer_0_params['mlp']
                    print(f"\nmlp keys: {list(mlp.keys())}")
                    
                    # Check all MLP projections
                    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                        if proj_name in mlp:
                            proj = mlp[proj_name]
                            print(f"{proj_name} type:", type(proj))
                            print(f"{proj_name} keys:", list(proj.keys()) if hasattr(proj, 'keys') else 'Not a dict')
                            if hasattr(proj, 'keys'):
                                for key, value in proj.items():
                                    print(f"  {proj_name}.{key}: shape {value.shape}")
        
        return params
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    params = inspect_jax_params() 