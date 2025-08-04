#!/usr/bin/env python3
"""
Test script for generation with tensor parallelism
"""
import os
import jax
import jax.numpy as jnp
import json

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

# Import after setting environment
from q25j7_tensor_parallel_fixed import setup_device_mesh, Qwen25ForCausalLM, generate_text

def test_generation():
    """Test generation with tensor parallelism"""
    
    # Setup mesh
    mesh = setup_device_mesh()
    print(f"Created mesh: {mesh}")
    
    # Create a simple config for testing
    config = {
        "vocab_size": 1000,  # Small vocab for testing
        "hidden_size": 512,  # Small model for testing
        "intermediate_size": 1024,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,  # GQA
        "head_dim": 64,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6
    }
    
    print("Config:", config)
    
    # Create model
    model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    print("Model created successfully")
    
    # Initialize parameters
    rng = jax.random.PRNGKey(0)
    batch_size, seq_len = 1, 5
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    params = model.init(rng, input_ids)
    
    print("Parameters initialized successfully")
    
    # Create a simple tokenizer mock
    class MockTokenizer:
        def __init__(self):
            self.eos_token_id = 999
        
        def encode(self, text, return_tensors="jax"):
            # Simple encoding: just return some token IDs
            return jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)
        
        def decode(self, token_id, skip_special_tokens=True):
            return f"token_{token_id}"
    
    tokenizer = MockTokenizer()
    
    # Test generation
    print("Testing generation...")
    with mesh:
        output, peak_mem, avg_time = generate_text(model, params, tokenizer, 3, "Hello")
    
    print(f"Generated output: {output}")
    print(f"Peak memory: {peak_mem:.2f} GB")
    print(f"Avg time per token: {avg_time:.4f} seconds")
    
    print("✅ Generation test passed!")

if __name__ == "__main__":
    test_generation() 