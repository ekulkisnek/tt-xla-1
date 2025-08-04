#!/usr/bin/env python3
"""
Simple debug to check weight shapes and logits
"""
import os
import jax
import jax.numpy as jnp
import json

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

from q25j7_tensor_parallel_fixed import setup_device_mesh, Qwen25ForCausalLM, load_params
from transformers import AutoTokenizer

def debug_model():
    """Debug the parallelized model"""
    
    # Setup
    mesh = setup_device_mesh()
    
    with open("weights/config.json") as f:
        config = json.load(f)
    
    print("=== PARALLELIZED MODEL DEBUG ===\n")
    
    # Create model
    model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("weights")
    
    # Initialize model
    rng = jax.random.PRNGKey(0)
    input_ids = jnp.ones((1, 10), dtype=jnp.int32)
    
    print("1. Initializing parameters...")
    init_params = model.init(rng, input_ids)
    
    print("2. Loading weights...")
    loaded_params = load_params(model, "weights", jnp.bfloat16)
    
    # Check weight shapes
    print("3. Checking weight shapes...")
    
    def check_weights(params, prefix=""):
        for key, value in params.items():
            if isinstance(value, dict):
                check_weights(value, f"{prefix}.{key}" if prefix else key)
            else:
                if hasattr(value, 'shape'):
                    print(f"{prefix}.{key}: {value.shape}")
    
    check_weights(loaded_params['params'])
    
    # Test with real input
    print("\n4. Testing with real input...")
    test_text = "What is 2 + 2?"
    input_ids = tokenizer.encode(test_text, return_tensors="jax")
    print(f"Input: '{test_text}' -> {input_ids.shape}")
    
    with mesh:
        outputs = model.apply(loaded_params, input_ids, return_dict=True)
        logits = outputs['logits']
        
    print(f"Logits shape: {logits.shape}")
    
    # Check last token logits
    last_logits = logits[0, -1, :]
    top_tokens = jnp.argsort(last_logits)[-10:][::-1]  # Top 10 tokens
    
    print("\n5. Top 10 predicted tokens:")
    for i, token_id in enumerate(top_tokens):
        token = tokenizer.decode(int(token_id))
        prob = float(jax.nn.softmax(last_logits)[token_id])
        print(f"  {i+1}. Token {token_id}: '{token}' (prob: {prob:.4f})")
    
    # Check if all logits are similar (indicating a problem)
    logit_std = float(jnp.std(last_logits))
    logit_mean = float(jnp.mean(last_logits))
    logit_max = float(jnp.max(last_logits))
    logit_min = float(jnp.min(last_logits))
    
    print(f"\n6. Logit statistics:")
    print(f"   Mean: {logit_mean:.4f}")
    print(f"   Std:  {logit_std:.4f}")
    print(f"   Max:  {logit_max:.4f}")
    print(f"   Min:  {logit_min:.4f}")
    
    if logit_std < 0.1:
        print("   ⚠️  WARNING: Very low logit variance - model may be broken!")
    else:
        print("   ✅ Logit variance looks normal")
    
    # Test generation
    print("\n7. Testing generation...")
    next_token = int(jnp.argmax(last_logits))
    next_token_text = tokenizer.decode(next_token)
    print(f"Next predicted token: {next_token} -> '{next_token_text}'")
    
    return True

if __name__ == "__main__":
    debug_model()