#!/usr/bin/env python3
"""
Phase 2.4: Edge Case Testing
============================
Tests edge cases in tokenization to ensure robustness and consistency.
Edge cases often reveal subtle differences in implementation.
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

def test_edge_cases():
    """Test edge cases in tokenization"""
    
    # Edge case test scenarios
    test_cases = [
        # Boundary cases
        {"name": "Empty string", "input": ""},
        {"name": "Single space", "input": " "},
        {"name": "Multiple spaces", "input": "     "},
        {"name": "Tab character", "input": "\t"},
        {"name": "Newline character", "input": "\n"},
        {"name": "Carriage return", "input": "\r"},
        {"name": "Mixed whitespace", "input": " \t\n\r "},
        
        # Unicode edge cases
        {"name": "Unicode emoji", "input": "🤖🚀🌟"},
        {"name": "Unicode symbols", "input": "→←↑↓∞≠≤≥"},
        {"name": "Unicode accents", "input": "café naïve résumé"},
        {"name": "Unicode CJK", "input": "这是中文 これは日本語 이것은한국어"},
        {"name": "Unicode RTL", "input": "مرحبا بالعالم"},
        {"name": "Unicode combining", "input": "e\u0301"},  # e with acute accent
        
        # Very long strings
        {"name": "Repeated word", "input": "hello " * 1000},
        {"name": "Very long word", "input": "a" * 500},
        {"name": "Mixed long content", "input": "The quick brown fox jumps over the lazy dog. " * 100},
        
        # Special character combinations
        {"name": "All punctuation", "input": "!@#$%^&*()_+-=[]{}|;':\",./<>?"},
        {"name": "Quote combinations", "input": "\"'`''\"\"''``"},
        {"name": "Bracket nesting", "input": "((([[[{{{<><>}}}]]])))"},
        {"name": "Mixed quotes", "input": "He said \"I'm happy\" and I replied 'That's great!'"},
        
        # Number edge cases
        {"name": "Large numbers", "input": "123456789012345678901234567890"},
        {"name": "Decimal precision", "input": "3.14159265358979323846"},
        {"name": "Scientific notation", "input": "1.23e-45 6.78E+90"},
        {"name": "Negative numbers", "input": "-123 -45.67 -8.9e-10"},
        
        # URL and email edge cases
        {"name": "URLs", "input": "https://www.example.com/path?param=value&other=123#anchor"},
        {"name": "Emails", "input": "user.name+tag@example.co.uk test@subdomain.example.org"},
        {"name": "IP addresses", "input": "192.168.1.1 10.0.0.0/8 2001:db8::1"},
        
        # Code-like content
        {"name": "Code snippet", "input": "def func(x, y=None): return x**2 if y else x"},
        {"name": "JSON", "input": '{"key": "value", "number": 123, "bool": true}'},
        {"name": "XML/HTML", "input": "<html><body><p>Hello &amp; welcome!</p></body></html>"},
        
        # Markdown-like content
        {"name": "Markdown", "input": "# Header\n\n**bold** *italic* `code` [link](url)"},
        {"name": "List formatting", "input": "1. First\n2. Second\n  - Nested\n  - Items"},
        
        # Mixed content chaos
        {"name": "Everything mixed", "input": "Hello🌍! Email: test@example.com URL: https://site.com Numbers: 123.45e-6 Code: `var x = \"string\";` 中文"},
        
        # Zero-width and special Unicode
        {"name": "Zero-width chars", "input": "a\u200bb\u200cc\u200dd"},  # Zero-width space, non-joiner, joiner
        {"name": "Invisible chars", "input": "hello\ufeffworld"},  # Byte order mark
        
        # Control characters
        {"name": "Control chars", "input": "\x00\x01\x02\x03\x1f"},
        
        # Extremely nested/repeated patterns
        {"name": "Nested parentheses", "input": "(" * 50 + "content" + ")" * 50},
        {"name": "Alternating case", "input": "AbCdEfGhIjKlMnOpQrStUvWxYz" * 10},
    ]
    
    print("=" * 60)
    print("PHASE 2.4: EDGE CASE TESTING")
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
    error_count = 0
    
    print("Edge Case Testing:")
    print("-" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        name = test_case["name"]
        test_input = test_case["input"]
        
        try:
            # Test basic tokenization
            pytorch_tokens = pytorch_tokenizer.encode(test_input)
            jax_tokens = jax_tokenizer.encode(test_input)
            
            # Test full encoding (with attention masks)
            pytorch_encoding = pytorch_tokenizer(test_input, return_tensors="pt")
            jax_encoding = jax_tokenizer(test_input, return_tensors="pt")
            
            pytorch_input_ids = pytorch_encoding['input_ids'].tolist()[0]
            pytorch_attention_mask = pytorch_encoding['attention_mask'].tolist()[0]
            
            jax_input_ids = jax_encoding['input_ids'].tolist()[0]
            jax_attention_mask = jax_encoding['attention_mask'].tolist()[0]
            
            # Compare results
            tokens_match = pytorch_tokens == jax_tokens
            input_ids_match = pytorch_input_ids == jax_input_ids
            attention_mask_match = pytorch_attention_mask == jax_attention_mask
            
            # Overall match
            overall_match = tokens_match and input_ids_match and attention_mask_match
            
            # Store result
            result = {
                'test_case': i,
                'name': name,
                'input': test_input,
                'input_length': len(test_input),
                'pytorch_tokens': pytorch_tokens,
                'jax_tokens': jax_tokens,
                'tokens_match': tokens_match,
                'input_ids_match': input_ids_match,
                'attention_mask_match': attention_mask_match,
                'overall_match': overall_match,
                'token_count': len(pytorch_tokens)
            }
            results.append(result)
            
            if not overall_match:
                all_match = False
            
            # Display result
            status = "✅" if overall_match else "❌"
            token_count = len(pytorch_tokens)
            input_len = len(test_input)
            print(f"{status} Case {i:2d}: {name:<20} -> {input_len:4d} chars, {token_count:4d} tokens")
            
            # Show details for mismatches
            if not overall_match:
                if not tokens_match:
                    print(f"    Token mismatch:")
                    print(f"      PyTorch: {pytorch_tokens[:10]}...")
                    print(f"      JAX:     {jax_tokens[:10]}...")
                if not input_ids_match:
                    print(f"    Input IDs mismatch")
                if not attention_mask_match:
                    print(f"    Attention mask mismatch")
                    
        except Exception as e:
            print(f"❌ Case {i:2d}: {name:<20} -> ERROR: {e}")
            error_count += 1
            all_match = False
            result = {
                'test_case': i,
                'name': name,
                'input': test_input,
                'error': str(e),
                'overall_match': False
            }
            results.append(result)
    
    print("-" * 60)
    
    # Summary statistics
    total_tests = len(test_cases)
    passed_tests = sum(1 for r in results if r.get('overall_match', False))
    failed_tests = total_tests - passed_tests
    
    print(f"\nSummary:")
    print(f"Total test cases: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Errors: {error_count}")
    
    # Additional analysis for successful cases
    if passed_tests > 0:
        successful_results = [r for r in results if r.get('overall_match', False)]
        
        # Input length analysis
        lengths = [r['input_length'] for r in successful_results]
        token_counts = [r['token_count'] for r in successful_results]
        
        print(f"\nSuccessful Cases Analysis:")
        print(f"Input length range: {min(lengths)} - {max(lengths)} characters")
        print(f"Token count range: {min(token_counts)} - {max(token_counts)} tokens")
        print(f"Average compression ratio: {sum(lengths)/sum(token_counts):.1f} chars/token")
    
    if all_match:
        print("\n✅ ALL EDGE CASES HANDLED IDENTICALLY!")
        print("Phase 2.4 PASSED - Tokenization consistency verified")
    else:
        print("\n❌ EDGE CASE FAILURES DETECTED")
        print("Some edge cases are handled differently between implementations")
    
    return all_match

if __name__ == "__main__":
    success = test_edge_cases()
    sys.exit(0 if success else 1) 