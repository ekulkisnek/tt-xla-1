#!/usr/bin/env python3
"""
Test weight loading for parallelized model
"""
import os
import jax
import jax.numpy as jnp
import json

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

from q25j7_tensor_parallel_fixed import setup_device_mesh, Qwen25ForCausalLM, load_params

def test_weight_loading():
    """Test weight loading for parallelized model"""
    
    # Setup
    mesh = setup_device_mesh()
    
    with open("weights/config.json") as f:
        config = json.load(f)
    
    print("=== TESTING WEIGHT LOADING ===")
    print(f"Config: {config['hidden_size']} hidden, {config['num_attention_heads']} heads")
    
    # Create model
    model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    
    # Initialize with random weights
    rng = jax.random.PRNGKey(0)
    input_ids = jnp.ones((1, 10), dtype=jnp.int32)
    
    print("1. Initializing with random weights...")
    init_params = model.init(rng, input_ids)
    
    # Test forward pass with random weights
    print("2. Testing forward pass with random weights...")
    with mesh:
        random_output = model.apply(init_params, input_ids, return_dict=True)
    
    print(f"Random weights output shape: {random_output['logits'].shape}")
    print(f"Random weights output min/max: {float(jnp.min(random_output['logits'])):.4f}, {float(jnp.max(random_output['logits'])):.4f}")
    
    # Load real weights
    print("3. Loading real weights...")
    loaded_params = load_params(model, "weights", jnp.bfloat16)
    
    # Test forward pass with loaded weights
    print("4. Testing forward pass with loaded weights...")
    with mesh:
        loaded_output = model.apply(loaded_params, input_ids, return_dict=True)
    
    print(f"Loaded weights output shape: {loaded_output['logits'].shape}")
    print(f"Loaded weights output min/max: {float(jnp.min(loaded_output['logits'])):.4f}, {float(jnp.max(loaded_output['logits'])):.4f}")
    
    # Compare outputs
    print("5. Comparing outputs...")
    random_logits = random_output['logits'][0, -1, :]  # Last token
    loaded_logits = loaded_output['logits'][0, -1, :]  # Last token
    
    print(f"Random logits mean/std: {float(jnp.mean(random_logits)):.4f}, {float(jnp.std(random_logits)):.4f}")
    print(f"Loaded logits mean/std: {float(jnp.mean(loaded_logits)):.4f}, {float(jnp.std(loaded_logits)):.4f}")
    
    # Check if loaded weights produce more reasonable outputs
    random_range = jnp.max(random_logits) - jnp.min(random_logits)
    loaded_range = jnp.max(loaded_logits) - jnp.min(loaded_logits)
    
    print(f"Random logits range: {float(random_range):.4f}")
    print(f"Loaded logits range: {float(loaded_range):.4f}")
    
    if loaded_range > random_range:
        print("✅ Loaded weights produce more varied outputs!")
    else:
        print("❌ Loaded weights don't seem to improve outputs")
    
    # Check top tokens
    random_top = jnp.argsort(random_logits)[-5:][::-1]
    loaded_top = jnp.argsort(loaded_logits)[-5:][::-1]
    
    print(f"Random top tokens: {random_top}")
    print(f"Loaded top tokens: {loaded_top}")
    
    return loaded_params, loaded_output

if __name__ == "__main__":
    test_weight_loading() 