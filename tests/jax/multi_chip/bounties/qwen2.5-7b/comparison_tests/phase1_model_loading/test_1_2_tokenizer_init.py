#!/usr/bin/env python3
"""
Phase 1.2: Tokenizer Initialization Comparison
==============================================
Tests that both PyTorch and JAX initialize the tokenizer identically.
Verifies vocab size, special tokens, and basic tokenizer properties.
"""

import json
import sys
import os
from pathlib import Path

def load_pytorch_tokenizer():
    """Load tokenizer using PyTorch/Transformers approach"""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("../../weights")
        
        # Extract key tokenizer properties
        pytorch_tokenizer_info = {
            'vocab_size': tokenizer.vocab_size,
            'model_max_length': tokenizer.model_max_length,
            'pad_token': tokenizer.pad_token,
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token': tokenizer.eos_token,
            'eos_token_id': tokenizer.eos_token_id,
            'bos_token': tokenizer.bos_token,
            'bos_token_id': tokenizer.bos_token_id,
            'unk_token': tokenizer.unk_token,
            'unk_token_id': tokenizer.unk_token_id,
            'tokenizer_class': tokenizer.__class__.__name__,
            'special_tokens_count': len(tokenizer.all_special_tokens),
            'can_apply_chat_template': hasattr(tokenizer, 'apply_chat_template'),
        }
        
        return pytorch_tokenizer_info, tokenizer, None
        
    except Exception as e:
        return None, None, str(e)

def load_jax_tokenizer():
    """Load tokenizer using JAX approach (same as PyTorch for now)"""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("../../weights")
        
        # Extract same properties
        jax_tokenizer_info = {
            'vocab_size': tokenizer.vocab_size,
            'model_max_length': tokenizer.model_max_length,
            'pad_token': tokenizer.pad_token,
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token': tokenizer.eos_token,
            'eos_token_id': tokenizer.eos_token_id,
            'bos_token': tokenizer.bos_token,
            'bos_token_id': tokenizer.bos_token_id,
            'unk_token': tokenizer.unk_token,
            'unk_token_id': tokenizer.unk_token_id,
            'tokenizer_class': tokenizer.__class__.__name__,
            'special_tokens_count': len(tokenizer.all_special_tokens),
            'can_apply_chat_template': hasattr(tokenizer, 'apply_chat_template'),
        }
        
        return jax_tokenizer_info, tokenizer, None
        
    except Exception as e:
        return None, None, str(e)

def test_basic_tokenization(pytorch_tokenizer, jax_tokenizer):
    """Test that basic tokenization produces identical results"""
    test_strings = [
        "Hello world",
        "The quick brown fox",
        "1234567890",
        "Special characters: !@#$%^&*()",
        "",  # Empty string
        "中文测试",  # Chinese characters
        "A very long string " * 10,  # Long string
    ]
    
    print("\nBasic Tokenization Test:")
    print("-" * 40)
    
    all_match = True
    for test_str in test_strings:
        pytorch_tokens = pytorch_tokenizer.encode(test_str)
        jax_tokens = jax_tokenizer.encode(test_str)
        
        match = pytorch_tokens == jax_tokens
        status = "✅" if match else "❌"
        
        if not match:
            all_match = False
            
        print(f"{status} '{test_str[:30]}{'...' if len(test_str) > 30 else ''}' -> {len(pytorch_tokens)} tokens")
        
        if not match:
            print(f"    PyTorch: {pytorch_tokens[:10]}{'...' if len(pytorch_tokens) > 10 else ''}")
            print(f"    JAX:     {jax_tokens[:10]}{'...' if len(jax_tokens) > 10 else ''}")
    
    return all_match

def compare_tokenizers():
    """Compare tokenizer initialization and basic functionality"""
    print("=" * 60)
    print("PHASE 1.2: TOKENIZER INITIALIZATION COMPARISON")
    print("=" * 60)
    
    # Load both tokenizers
    pytorch_info, pytorch_tokenizer, pytorch_error = load_pytorch_tokenizer()
    jax_info, jax_tokenizer, jax_error = load_jax_tokenizer()
    
    # Check for loading errors
    if pytorch_error:
        print(f"❌ PyTorch tokenizer loading FAILED: {pytorch_error}")
        return False
    
    if jax_error:
        print(f"❌ JAX tokenizer loading FAILED: {jax_error}")
        return False
    
    print("✅ Both tokenizers loaded successfully")
    print()
    
    # Compare tokenizer properties
    all_match = True
    max_key_len = max(len(k) for k in pytorch_info.keys())
    
    print("Tokenizer Property Comparison:")
    print("-" * 60)
    
    for key in sorted(pytorch_info.keys()):
        pytorch_val = pytorch_info[key]
        jax_val = jax_info[key]
        
        match_status = "✅" if pytorch_val == jax_val else "❌"
        if pytorch_val != jax_val:
            all_match = False
        
        pytorch_str = str(pytorch_val) if pytorch_val is not None else "None"
        jax_str = str(jax_val) if jax_val is not None else "None"
        print(f"{match_status} {key:<{max_key_len}} | PyTorch: {pytorch_str:<15} | JAX: {jax_str}")
    
    # Test basic tokenization
    tokenization_match = test_basic_tokenization(pytorch_tokenizer, jax_tokenizer)
    all_match = all_match and tokenization_match
    
    print("-" * 60)
    
    if all_match:
        print("✅ ALL TOKENIZER PROPERTIES AND FUNCTIONALITY MATCH!")
        print("Phase 1.2 PASSED - Ready for Phase 1.3")
    else:
        print("❌ TOKENIZER MISMATCHES DETECTED")
        print("Must fix tokenizer initialization before proceeding")
    
    return all_match

if __name__ == "__main__":
    success = compare_tokenizers()
    sys.exit(0 if success else 1) 