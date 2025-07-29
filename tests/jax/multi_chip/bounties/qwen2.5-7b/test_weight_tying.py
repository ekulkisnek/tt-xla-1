#!/usr/bin/env python3
"""
Quick test to verify weight tying behavior
"""

import os
import json
import numpy as np
import jax.numpy as jnp
from qwen_jax_inference import Qwen25ForCausalLM, load_params

def test_weight_tying():
    """Test if embedding and lm_head weights are properly handled"""
    print("🔍 Testing weight tying behavior...")
    
    # Load config
    config_path = "./weights/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Config tie_word_embeddings: {config.get('tie_word_embeddings', 'not found')}")
    
    # Load model and params
    model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
    params = load_params(model, "./weights", jnp.float32)
    
    # Get embedding and lm_head weights
    embed_weight = params['params']['embed_tokens']['embedding']
    lm_head_weight = params['params']['lm_head']['kernel']
    
    print(f"Embedding weight shape: {embed_weight.shape}")
    print(f"LM head weight shape: {lm_head_weight.shape}")
    
    # Check if they're tied (should NOT be since tie_word_embeddings=false)
    if embed_weight.shape == lm_head_weight.shape:
        # Check if they're the same
        max_diff = jnp.max(jnp.abs(embed_weight - lm_head_weight))
        print(f"Max difference between embed and lm_head: {float(max_diff):.2e}")
        
        if float(max_diff) < 1e-6:
            print("❌ ERROR: Weights are tied but should NOT be (tie_word_embeddings=false)")
            return False
        else:
            print("✅ Weights are NOT tied (correct)")
            return True
    elif embed_weight.shape == lm_head_weight.T.shape:
        # Check if lm_head is transpose of embedding
        max_diff = jnp.max(jnp.abs(embed_weight - lm_head_weight.T))
        print(f"Max difference between embed and lm_head.T: {float(max_diff):.2e}")
        
        if float(max_diff) < 1e-6:
            print("❌ ERROR: LM head is transpose of embedding (incorrect tying)")
            return False
        else:
            print("✅ Weights are different (correct)")
            return True
    else:
        print("❌ ERROR: Embedding and LM head have incompatible shapes!")
        return False

def test_simple_forward_pass():
    """Test a simple forward pass to identify issues"""
    print("\n🔍 Testing simple forward pass...")
    
    # Load config
    config_path = "./weights/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load model and params  
    model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
    params = load_params(model, "./weights", jnp.float32)
    
    # Test with a simple input
    input_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    outputs = model.apply(params, input_ids=input_ids)
    if isinstance(outputs, dict):
        logits = outputs["logits"]
    else:
        logits = outputs
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Logits range: [{float(jnp.min(logits)):.3f}, {float(jnp.max(logits)):.3f}]")
    print(f"Logits mean: {float(jnp.mean(logits)):.3f}")
    print(f"Logits std: {float(jnp.std(logits)):.3f}")
    
    # Check for NaN/Inf
    if jnp.any(jnp.isnan(logits)) or jnp.any(jnp.isinf(logits)):
        print("❌ ERROR: NaN or Inf values in logits!")
        return False
    else:
        print("✅ No NaN/Inf values")
        return True

if __name__ == "__main__":
    print("🚀 Weight Tying and Forward Pass Test")
    print("="*50)
    
    # Test weight tying
    tying_ok = test_weight_tying()
    
    # Test forward pass
    forward_ok = test_simple_forward_pass()
    
    print("\n" + "="*50)
    print("🎯 SUMMARY")
    print("="*50)
    print(f"Weight tying correct: {'✅' if tying_ok else '❌'}")
    print(f"Forward pass works: {'✅' if forward_ok else '❌'}")
    
    if tying_ok and forward_ok:
        print("✅ Basic tests passed - issue is likely in detailed implementation")
    else:
        print("❌ Found fundamental issues - fix these first") 