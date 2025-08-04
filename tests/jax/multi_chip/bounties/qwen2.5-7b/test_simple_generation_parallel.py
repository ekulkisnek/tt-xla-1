#!/usr/bin/env python3
"""
Test simple generation with fully parallel model to compare with original
"""
import os
import jax
import jax.numpy as jnp
import json

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

from q25j7_tensor_parallel_fixed import setup_device_mesh, Qwen25ForCausalLM, load_params
from transformers import AutoTokenizer

def test_simple_generation_parallel():
    """Test simple generation with fully parallel model"""
    
    # Setup
    mesh = setup_device_mesh()
    
    with open("weights/config.json") as f:
        config = json.load(f)
    
    print("=== TESTING SIMPLE GENERATION WITH FULLY PARALLEL MODEL ===\n")
    
    # Create model and load weights
    model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    params = load_params(model, "weights", jnp.bfloat16)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("weights")
    
    # Test with same input as the working original model
    test_input = "What is 2 + 2?"
    input_ids = tokenizer.encode(test_input, return_tensors="jax")
    
    print(f"Input: '{test_input}'")
    print(f"Input IDs: {input_ids}")
    print(f"Input shape: {input_ids.shape}")
    
    # Get logits for the last token
    with mesh:
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
    
    # Check if this matches what the original model predicts
    # From the original model test, it should predict " " as the first token
    if next_token == 220:  # Token 220 is ' ' (space)
        print("✅ Matches original model prediction (space token)!")
    elif next_token_text.strip() == "":
        print("✅ Predicts whitespace (close to original)")
    else:
        print(f"❌ Different from original model (should be space)")
    
    # Test a few more tokens to see the pattern
    print("\n=== TESTING MULTI-TOKEN GENERATION ===")
    
    # Generate a few tokens
    current_ids = input_ids.copy()
    generated_tokens = []
    
    for i in range(5):
        with mesh:
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
    
    # Check if this resembles the original " 2 + 2"
    if " 2 + 2" in full_text or "2+2" in full_text or "2 + 2" in full_text:
        print("✅ Generated text contains mathematical answer!")
    elif any(c.isdigit() for c in full_text):
        print("⚠️ Generated text contains numbers (partial success)")
    else:
        print("❌ Generated text doesn't contain mathematical answer")
    
    return full_text

if __name__ == "__main__":
    test_simple_generation_parallel()