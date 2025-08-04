#!/usr/bin/env python3
"""
Test script for model initialization with tensor parallelism
"""
import os
import jax
import jax.numpy as jnp
import json

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

# Import after setting environment
from q25j7_tensor_parallel_fixed import setup_device_mesh, Qwen25ForCausalLM

def test_model_init():
    """Test model initialization with tensor parallelism"""
    
    # Setup mesh
    mesh = setup_device_mesh()
    print(f"Created mesh: {mesh}")
    
    # Create a simple config for testing
    config = {
        "vocab_size": 151936,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 2,  # Small for testing
        "num_attention_heads": 32,
        "num_key_value_heads": 8,  # GQA
        "head_dim": 128,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6
    }
    
    print("Config:", config)
    
    # Create model
    model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    print("Model created successfully")
    
    # Create test input
    batch_size, seq_len = 1, 10
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    
    print(f"Input shape: {input_ids.shape}")
    
    # Initialize parameters
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, input_ids)
    
    print("Parameters initialized successfully")
    
    # Test forward pass
    with mesh:
        outputs = model.apply(params, input_ids, return_dict=True)
    
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Expected logits shape: {input_ids.shape + (config['vocab_size'],)}")
    
    assert outputs['logits'].shape == (batch_size, seq_len, config['vocab_size']), \
        f"Expected {(batch_size, seq_len, config['vocab_size'])}, got {outputs['logits'].shape}"
    
    print("✅ Model initialization test passed!")

if __name__ == "__main__":
    test_model_init() 