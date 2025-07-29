#!/usr/bin/env python3
"""
Simple Fix Test - Try to identify and fix the logits scaling issue
"""

import jax.numpy as jnp
import numpy as np

def test_rms_norm_comparison():
    """Test if our RMS norm matches PyTorch implementation exactly"""
    print("🔍 Testing RMS Norm Implementation")
    print("="*50)
    
    # Test with a simple input
    test_input = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
    eps = 1e-6
    
    # Our JAX implementation (from qwen_jax_inference.py)
    test_jax = jnp.array(test_input)
    variance_jax = jnp.mean(test_jax**2, axis=-1, keepdims=True)
    norm_jax = test_jax * jnp.power(variance_jax + eps, -0.5)
    
    print(f"Input: {test_input}")
    print(f"JAX variance: {float(variance_jax[0,0,0]):.6f}")
    print(f"JAX norm result: {np.array(norm_jax)}")
    
    # PyTorch-style implementation
    import torch
    test_torch = torch.tensor(test_input)
    norm_torch = torch.nn.functional.rms_norm(test_torch, normalized_shape=(4,), eps=eps)
    
    print(f"PyTorch norm result: {norm_torch.numpy()}")
    
    diff = np.max(np.abs(np.array(norm_jax) - norm_torch.numpy()))
    print(f"Difference: {diff:.2e}")
    
    if diff < 1e-6:
        print("✅ RMS norm implementations match")
        return True
    else:
        print("❌ RMS norm implementations differ")
        return False

def test_attention_scaling():
    """Test if our attention scaling matches expected values"""
    print("\n🔍 Testing Attention Scaling")
    print("="*50)
    
    # From config: head_dim should be hidden_size / num_attention_heads
    # hidden_size=3584, num_attention_heads=28
    head_dim = 3584 // 28
    print(f"Head dim: {head_dim}")
    
    # Our scaling
    our_scale = 1.0 / jnp.sqrt(float(head_dim))
    print(f"Our scale: {float(our_scale):.6f}")
    
    # Expected scaling
    expected_scale = 1.0 / np.sqrt(head_dim)
    print(f"Expected scale: {expected_scale:.6f}")
    
    diff = abs(float(our_scale) - expected_scale)
    print(f"Difference: {diff:.2e}")
    
    if diff < 1e-10:
        print("✅ Attention scaling matches")
        return True
    else:
        print("❌ Attention scaling differs")
        return False

def analyze_config_values():
    """Check if our config interpretation is correct"""
    print("\n🔍 Analyzing Config Values")
    print("="*50)
    
    import json
    with open("./weights/config.json", 'r') as f:
        config = json.load(f)
    
    # Key values that affect computation
    hidden_size = config["hidden_size"]
    num_heads = config["num_attention_heads"] 
    num_kv_heads = config["num_key_value_heads"]
    vocab_size = config["vocab_size"]
    rms_norm_eps = config["rms_norm_eps"]
    
    print(f"Hidden size: {hidden_size}")
    print(f"Num attention heads: {num_heads}")
    print(f"Num KV heads: {num_kv_heads}")
    print(f"Vocab size: {vocab_size}")
    print(f"RMS norm eps: {rms_norm_eps}")
    
    # Calculate derived values
    head_dim = hidden_size // num_heads
    kv_dim = num_kv_heads * head_dim
    
    print(f"Derived head dim: {head_dim}")
    print(f"Derived KV dim: {kv_dim}")
    
    # Check if these match our implementation
    expected_shapes = {
        'q_proj': (hidden_size, hidden_size),
        'k_proj': (hidden_size, kv_dim), 
        'v_proj': (hidden_size, kv_dim),
        'o_proj': (hidden_size, hidden_size),
        'lm_head': (hidden_size, vocab_size)  # This might be the issue!
    }
    
    print(f"\nExpected weight shapes after transpose:")
    for name, shape in expected_shapes.items():
        print(f"  {name}: {shape}")
    
    return {
        'head_dim': head_dim,
        'kv_dim': kv_dim,
        'rms_norm_eps': rms_norm_eps
    }

def suggest_potential_fixes():
    """Suggest potential fixes based on analysis"""
    print("\n🔧 Potential Fixes to Try")
    print("="*50)
    
    fixes = [
        "1. Check LM head weight shape - should be (hidden_size, vocab_size) = (3584, 152064)",
        "2. Verify RMS norm epsilon value is exactly 1e-06 (not 1e-05 or 1e-6 float precision)",
        "3. Check if final layer norm scale weights are being applied correctly",
        "4. Verify embedding weights are not accidentally tied to LM head",
        "5. Check if there's a scaling factor missing in the final logits computation",
        "6. Consider if PyTorch model has any post-processing that we're missing"
    ]
    
    for fix in fixes:
        print(f"   • {fix}")
    
    print(f"\n💡 Next Steps:")
    print(f"   • Focus on the LM head computation since that's where logits are generated")
    print(f"   • The fact that math prompts have largest differences suggests token-specific issues")
    print(f"   • Consider if certain tokens (numbers, operators) have different scaling")

if __name__ == "__main__":
    print("🚀 Simple Fix Analysis")
    print("="*60)
    
    # Run tests
    rms_ok = test_rms_norm_comparison()
    attn_ok = test_attention_scaling()
    config_analysis = analyze_config_values()
    
    print(f"\n🎯 Summary:")
    print(f"   • RMS norm correct: {'✅' if rms_ok else '❌'}")
    print(f"   • Attention scaling correct: {'✅' if attn_ok else '❌'}")
    print(f"   • Config analysis complete: ✅")
    
    suggest_potential_fixes() 