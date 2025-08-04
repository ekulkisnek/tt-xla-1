#!/usr/bin/env python3
"""
Debug script to check weight loading and shapes
"""
import os
import jax
import jax.numpy as jnp
import json

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

from q25j7_tensor_parallel_fixed import setup_device_mesh, Qwen25ForCausalLM, load_params
from q25j7_tensor_parallel import Qwen25ForCausalLM as OriginalModel, load_params as original_load_params

def debug_weights():
    """Debug weight loading differences"""
    
    # Setup
    mesh = setup_device_mesh()
    
    with open("weights/config.json") as f:
        config = json.load(f)
    
    print("=== WEIGHT LOADING DEBUG ===\n")
    
    # Create both models
    print("1. Creating models...")
    parallel_model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    original_model = OriginalModel(config=config, dtype=jnp.bfloat16)
    
    # Initialize both models
    rng = jax.random.PRNGKey(0)
    input_ids = jnp.ones((1, 10), dtype=jnp.int32)
    
    print("2. Initializing parameters...")
    parallel_params = parallel_model.init(rng, input_ids)
    original_params = original_model.init(rng, input_ids)
    
    print("3. Checking parameter structure differences...")
    
    def print_param_structure(params, name, level=0):
        indent = "  " * level
        if isinstance(params, dict):
            for key, value in params.items():
                if isinstance(value, dict):
                    print(f"{indent}{key}/")
                    print_param_structure(value, f"{name}.{key}", level + 1)
                else:
                    if hasattr(value, 'shape'):
                        print(f"{indent}{key}: {value.shape}")
                    else:
                        print(f"{indent}{key}: {type(value)}")
        else:
            if hasattr(params, 'shape'):
                print(f"{indent}{name}: {params.shape}")
            else:
                print(f"{indent}{name}: {type(params)}")
    
    print("\n--- Original Model Structure ---")
    print_param_structure(original_params['params'], "original")
    
    print("\n--- Parallel Model Structure ---")
    print_param_structure(parallel_params['params'], "parallel")
    
    # Load weights
    print("\n4. Loading weights...")
    loaded_parallel = load_params(parallel_model, "weights", jnp.bfloat16)
    loaded_original = original_load_params(original_model, "weights", jnp.bfloat16)
    
    # Compare specific weights
    print("\n5. Comparing key weights...")
    
    # Check embedding
    orig_embed = loaded_original['params']['embed_tokens']['embedding']
    par_embed = loaded_parallel['params']['embed_tokens']['embedding']
    print(f"Embedding - Original: {orig_embed.shape}, Parallel: {par_embed.shape}")
    print(f"Embedding equal: {jnp.allclose(orig_embed, par_embed)}")
    
    # Check first layer attention weights
    layer_0_orig = loaded_original['params']['layers_0']['self_attn']
    layer_0_par = loaded_parallel['params']['layers_0']['self_attn']
    
    for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        if proj in layer_0_orig and proj in layer_0_par:
            orig_weight = layer_0_orig[proj]['kernel']
            par_weight = layer_0_par[proj]['kernel']
            print(f"{proj} - Original: {orig_weight.shape}, Parallel: {par_weight.shape}")
            print(f"{proj} equal: {jnp.allclose(orig_weight, par_weight)}")
    
    # Test forward pass with same inputs
    print("\n6. Testing forward passes...")
    
    with mesh:
        # Test parallel model
        par_output = parallel_model.apply(loaded_parallel, input_ids, return_dict=True)
        par_logits = par_output['logits'][0, -1, :]  # Last token logits
        
        # Test original model  
        orig_output = original_model.apply(loaded_original, input_ids, return_dict=True)
        orig_logits = orig_output['logits'][0, -1, :]  # Last token logits
    
    print(f"Original top token: {jnp.argmax(orig_logits)}")
    print(f"Parallel top token: {jnp.argmax(par_logits)}")
    print(f"Logits equal: {jnp.allclose(orig_logits, par_logits, rtol=1e-3)}")
    print(f"Max diff: {jnp.max(jnp.abs(orig_logits - par_logits))}")
    
    return True

if __name__ == "__main__":
    debug_weights()