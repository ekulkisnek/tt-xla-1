#!/usr/bin/env python3
"""
Debug weight shape mismatches between original and parallel models
"""
import os
import jax
import jax.numpy as jnp
import json

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

from q25j7_tensor_parallel_fixed import Qwen25ForCausalLM as ParallelModel, setup_device_mesh

def debug_weight_shapes():
    """Debug weight shapes for parallel model"""
    
    # Setup
    mesh = setup_device_mesh()
    
    with open("weights/config.json") as f:
        config = json.load(f)
    
    print("=== WEIGHT SHAPE DEBUG ===\n")
    print(f"Config: {config['hidden_size']} hidden, {config['num_attention_heads']} heads, {config['num_key_value_heads']} kv_heads")
    
    # Create parallel model
    model = ParallelModel(config=config, dtype=jnp.bfloat16)
    
    # Initialize model to see expected shapes
    rng = jax.random.PRNGKey(0)
    input_ids = jnp.ones((1, 10), dtype=jnp.int32)
    
    print("1. Checking initialized parameter shapes...")
    params = model.init(rng, input_ids)
    
    def print_shapes(params, prefix=""):
        for key, value in params.items():
            if isinstance(value, dict):
                print_shapes(value, f"{prefix}.{key}" if prefix else key)
            else:
                if hasattr(value, 'shape'):
                    print(f"{prefix}.{key}: {value.shape}")
    
    print_shapes(params['params'])
    
    print("\n2. Expected vs Actual for key layers:")
    
    # Expected shapes for 28 heads, 4 kv heads, 3584 hidden
    expected_shapes = {
        "q_proj": (3584, 3584),  # query: hidden -> num_heads * head_dim = 28 * 128 = 3584
        "k_proj": (3584, 512),   # key: hidden -> num_kv_heads * head_dim = 4 * 128 = 512  
        "v_proj": (3584, 512),   # value: hidden -> num_kv_heads * head_dim = 4 * 128 = 512
        "o_proj": (3584, 3584),  # output: hidden -> hidden
        "gate_proj": (3584, 18944),  # gate: hidden -> intermediate
        "up_proj": (3584, 18944),    # up: hidden -> intermediate
        "down_proj": (18944, 3584),  # down: intermediate -> hidden
        "lm_head": (3584, 152064),   # lm_head: hidden -> vocab
    }
    
    # Check if ParallelDense changes these shapes
    layer_0 = params['params']['layers_0']
    actual_shapes = {}
    
    # Attention shapes
    for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        if proj in layer_0['self_attn']:
            actual_shapes[proj] = layer_0['self_attn'][proj]['kernel'].shape
    
    # MLP shapes  
    for proj in ['gate_proj', 'up_proj', 'down_proj']:
        if proj in layer_0['mlp']:
            actual_shapes[proj] = layer_0['mlp'][proj]['kernel'].shape
    
    # LM head
    actual_shapes['lm_head'] = params['params']['lm_head']['kernel'].shape
    
    print("Expected vs Actual shapes:")
    for layer, expected in expected_shapes.items():
        actual = actual_shapes.get(layer, "Missing")
        match = "✅" if actual == expected else "❌"
        print(f"  {layer}: Expected {expected}, Got {actual} {match}")
        
        if actual != expected and actual != "Missing":
            print(f"    ⚠️  MISMATCH! This will cause incorrect behavior")
    
    print(f"\n3. Key findings:")
    print(f"   - Model expects specific shapes based on GQA config")
    print(f"   - ParallelDense should maintain same output shapes")
    print(f"   - Weight loading should match these exact shapes")
    
    print(f"\n4. Recommendations:")
    print(f"   - Verify ParallelDense doesn't change expected output dimensions")
    print(f"   - Check if weights are being sharded when they shouldn't be")
    print(f"   - Ensure weight loading matches expected parameter structure")

if __name__ == "__main__":
    debug_weight_shapes()