#!/usr/bin/env python3
"""
Test weight loading with parallel attention to identify issues
"""
import os
import jax
import jax.numpy as jnp
import json

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

from q25j7_tensor_parallel_fixed import setup_device_mesh, Qwen25ForCausalLM, load_params

def test_weight_loading_parallel_attention():
    """Test weight loading with fully parallel attention"""
    
    # Setup
    mesh = setup_device_mesh()
    
    with open("weights/config.json") as f:
        config = json.load(f)
    
    print("=== TESTING WEIGHT LOADING WITH PARALLEL ATTENTION ===\n")
    
    # Create model with fully parallel attention
    model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    
    # Initialize with random weights first
    rng = jax.random.PRNGKey(0)
    input_ids = jnp.ones((1, 10), dtype=jnp.int32)
    
    print("1. Checking model structure with parallel attention...")
    init_params = model.init(rng, input_ids)
    
    # Check attention parameter structure
    print("\nAttention parameter structure:")
    attention_params = init_params['params']['layers_0']['self_attn']
    for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        if proj in attention_params:
            if 'kernel' in attention_params[proj]:
                shape = attention_params[proj]['kernel'].shape
                print(f"  {proj}: {shape} ✅ (ParallelDense)")
            else:
                print(f"  {proj}: {list(attention_params[proj].keys())} ❌ (nn.Dense)")
        else:
            print(f"  {proj}: missing!")
    
    print("\n2. Testing forward pass with random weights...")
    try:
        with mesh:
            random_output = model.apply(init_params, input_ids, return_dict=True)
        
        print(f"Random output shape: {random_output['logits'].shape}")
        random_logits = random_output['logits'][0, -1, :]
        print(f"Random logits range: {float(jnp.min(random_logits)):.4f} to {float(jnp.max(random_logits)):.4f}")
        random_top_token = int(jnp.argmax(random_logits))
        print(f"Random top token: {random_top_token}")
        print("✅ Random weights work!")
        
    except Exception as e:
        print(f"❌ Random weights failed: {e}")
        return False
    
    print("\n3. Loading real weights...")
    try:
        loaded_params = load_params(model, "weights", jnp.bfloat16)
        print("✅ Weight loading completed")
        
        # Check if weights were loaded correctly for attention
        attention_loaded = loaded_params['params']['layers_0']['self_attn']
        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if proj in attention_loaded and 'kernel' in attention_loaded[proj]:
                shape = attention_loaded[proj]['kernel'].shape
                print(f"  Loaded {proj}: {shape}")
            else:
                print(f"  ❌ {proj} not loaded correctly")
        
    except Exception as e:
        print(f"❌ Weight loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n4. Testing forward pass with loaded weights...")
    try:
        with mesh:
            loaded_output = model.apply(loaded_params, input_ids, return_dict=True)
        
        print(f"Loaded output shape: {loaded_output['logits'].shape}")
        loaded_logits = loaded_output['logits'][0, -1, :]
        print(f"Loaded logits range: {float(jnp.min(loaded_logits)):.4f} to {float(jnp.max(loaded_logits)):.4f}")
        loaded_top_token = int(jnp.argmax(loaded_logits))
        print(f"Loaded top token: {loaded_top_token}")
        
        # Check if the outputs are different (they should be)
        if random_top_token != loaded_top_token:
            print("✅ Loaded weights produce different outputs from random")
        else:
            print("❌ Loaded weights produce same outputs as random")
        
        # Check if output is reasonable
        if float(jnp.std(loaded_logits)) > 1.0:
            print("✅ Loaded weights produce reasonable variance")
        else:
            print("❌ Loaded weights produce low variance (possible issue)")
        
        return True
        
    except Exception as e:
        print(f"❌ Loaded weights failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_weight_loading_parallel_attention()