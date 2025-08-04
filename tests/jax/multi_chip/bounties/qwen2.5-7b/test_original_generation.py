#!/usr/bin/env python3
"""
Test generation with original non-parallelized model for comparison
"""
import os
import jax
import jax.numpy as jnp
import json

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

from q25j7_tensor_parallel import Qwen25ForCausalLM as OriginalModel, load_params as original_load_params
from transformers import AutoTokenizer

def test_original_generation():
    """Test generation with original model"""
    
    print("=== TESTING ORIGINAL MODEL ===")
    
    with open("weights/config.json") as f:
        config = json.load(f)
    
    # Create original model and load weights
    model = OriginalModel(config=config, dtype=jnp.bfloat16)
    params = original_load_params(model, "weights", jnp.bfloat16)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("weights")
    
    # Test with same input
    test_input = "Hello"
    input_ids = tokenizer.encode(test_input, return_tensors="jax")
    
    print(f"Input: '{test_input}'")
    print(f"Input IDs: {input_ids}")
    print(f"Input shape: {input_ids.shape}")
    
    # Get logits for the last token
    outputs = model.apply(params, input_ids, return_dict=True)
    logits = outputs['logits']
    
    print(f"Logits shape: {logits.shape}")
    
    # Get last token logits
    last_logits = logits[0, -1, :]
    
    print(f"Last token logits shape: {last_logits.shape}")
    print(f"Last token logits min/max: {float(jnp.min(last_logits)):.4f}, {float(jnp.max(last_logits)):.4f}")
    print(f"Last token logits mean/std: {float(jnp.mean(last_logits)):.4f}, {float(jnp.std(last_logits)):.4f}")
    
    # Get top tokens
    top_tokens = jnp.argsort(last_logits)[-10:][::-1]
    top_probs = jax.nn.softmax(last_logits)[top_tokens]
    
    print("\nTop 10 predicted tokens:")
    for i, (token_id, prob) in enumerate(zip(top_tokens, top_probs)):
        token_text = tokenizer.decode(int(token_id))
        print(f"  {i+1}. Token {token_id}: '{token_text}' (prob: {float(prob):.4f})")
    
    # Test next token prediction
    next_token = int(jnp.argmax(last_logits))
    next_token_text = tokenizer.decode(next_token)
    
    print(f"\nPredicted next token: {next_token} -> '{next_token_text}'")
    
    # Test with a few more tokens
    print("\n=== TESTING MULTI-TOKEN GENERATION ===")
    
    # Generate a few tokens
    current_ids = input_ids.copy()
    generated_tokens = []
    
    for i in range(5):
        outputs = model.apply(params, current_ids, return_dict=True)
        logits = outputs['logits']
        
        next_logits = logits[0, -1, :]
        next_token = int(jnp.argmax(next_logits))
        next_text = tokenizer.decode(next_token)
        
        generated_tokens.append(next_token)
        current_ids = jnp.concatenate([current_ids, jnp.array([[next_token]])], axis=1)
        
        print(f"Step {i+1}: Token {next_token} -> '{next_text}'")
    
    # Decode full sequence
    full_text = tokenizer.decode(current_ids[0])
    print(f"\nFull generated text: '{full_text}'")
    
    return full_text

if __name__ == "__main__":
    test_original_generation() 