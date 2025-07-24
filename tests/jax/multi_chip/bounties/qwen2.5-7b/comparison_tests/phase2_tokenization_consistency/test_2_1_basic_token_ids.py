#!/usr/bin/env python3
"""
Phase 2.1: Basic Token ID Comparison
====================================
Tests that both PyTorch and JAX tokenizers produce identical token IDs for basic inputs.
This is the foundation test to ensure tokenization is identical.
"""

import sys
from pathlib import Path

def load_pytorch_tokenizer():
    """Load tokenizer using PyTorch/Transformers approach"""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("../../weights")
        return tokenizer, None
    except Exception as e:
        return None, str(e)

def load_jax_tokenizer():
    """Load tokenizer using JAX approach (same as PyTorch for tokenization)"""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("../../weights")
        return tokenizer, None
    except Exception as e:
        return None, str(e)

def test_basic_tokenization():
    """Test basic tokenization with various input types"""
    
    # Test strings covering different scenarios
    test_cases = [
        # Basic text
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        
        # Single words
        "hello",
        "world",
        "tokenization",
        
        # Numbers
        "123",
        "456789",
        "3.14159",
        
        # Special characters
        "!@#$%^&*()",
        "Hello, world!",
        "What's happening?",
        
        # Mixed content
        "Hello 123 world!",
        "Test-case_example",
        
        # Different languages (basic)
        "中文",
        "français",
        "español",
        
        # Edge cases
        "",  # Empty string
        " ",  # Single space
        "   ",  # Multiple spaces
        "\n",  # Newline
        "\t",  # Tab
    ]
    
    print("=" * 60)
    print("PHASE 2.1: BASIC TOKEN ID COMPARISON")
    print("=" * 60)
    
    # Load tokenizers
    pytorch_tokenizer, pytorch_error = load_pytorch_tokenizer()
    jax_tokenizer, jax_error = load_jax_tokenizer()
    
    if pytorch_error:
        print(f"❌ PyTorch tokenizer loading FAILED: {pytorch_error}")
        return False
    
    if jax_error:
        print(f"❌ JAX tokenizer loading FAILED: {jax_error}")
        return False
    
    print("✅ Both tokenizers loaded successfully")
    print()
    
    # Test each case
    all_match = True
    results = []
    
    print("Token ID Comparison:")
    print("-" * 60)
    
    for i, test_string in enumerate(test_cases, 1):
        # Tokenize with both
        pytorch_tokens = pytorch_tokenizer.encode(test_string)
        jax_tokens = jax_tokenizer.encode(test_string)
        
        # Compare
        tokens_match = pytorch_tokens == jax_tokens
        
        # Store result
        result = {
            'test_case': i,
            'input': test_string,
            'pytorch_tokens': pytorch_tokens,
            'jax_tokens': jax_tokens,
            'match': tokens_match,
            'token_count': len(pytorch_tokens)
        }
        results.append(result)
        
        if not tokens_match:
            all_match = False
        
        # Display result
        status = "✅" if tokens_match else "❌"
        input_display = repr(test_string) if len(test_string) <= 30 else repr(test_string[:27] + "...")
        print(f"{status} Case {i:2d}: {input_display:<35} -> {len(pytorch_tokens)} tokens")
        
        # Show details for mismatches
        if not tokens_match:
            print(f"    PyTorch: {pytorch_tokens}")
            print(f"    JAX:     {jax_tokens}")
    
    print("-" * 60)
    
    # Summary
    total_tests = len(test_cases)
    passed_tests = sum(1 for r in results if r['match'])
    
    print(f"\nSummary:")
    print(f"Total test cases: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    
    if all_match:
        print("✅ ALL TOKEN IDS MATCH!")
        print("Phase 2.1 PASSED - Ready for Phase 2.2")
    else:
        print("❌ TOKEN ID MISMATCHES DETECTED")
        print("Must fix tokenization before proceeding")
    
    return all_match

if __name__ == "__main__":
    success = test_basic_tokenization()
    sys.exit(0 if success else 1) 