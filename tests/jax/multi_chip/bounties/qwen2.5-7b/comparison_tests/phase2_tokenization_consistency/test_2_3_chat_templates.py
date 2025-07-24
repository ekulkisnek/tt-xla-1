#!/usr/bin/env python3
"""
Phase 2.3: Chat Template Application
===================================
Tests that both PyTorch and JAX tokenizers apply chat templates identically.
Chat templates are crucial for proper conversation formatting in instruct models.
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

def test_chat_templates():
    """Test chat template application with various conversation scenarios"""
    
    # Test cases covering different chat template scenarios
    test_cases = [
        # Simple single-turn conversation
        {
            "name": "Simple Q&A",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ]
        },
        
        # Multi-turn conversation
        {
            "name": "Multi-turn",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
                {"role": "user", "content": "What about 3+3?"}
            ]
        },
        
        # No system message
        {
            "name": "No system",
            "messages": [
                {"role": "user", "content": "Tell me a joke."}
            ]
        },
        
        # Different system messages
        {
            "name": "Math tutor",
            "messages": [
                {"role": "system", "content": "You are a math tutor. Explain concepts clearly."},
                {"role": "user", "content": "Explain calculus."}
            ]
        },
        
        # Long messages
        {
            "name": "Long messages",
            "messages": [
                {"role": "system", "content": "You are an expert in artificial intelligence with deep knowledge of machine learning, neural networks, and natural language processing."},
                {"role": "user", "content": "Can you explain how transformer models work, including the attention mechanism, positional encodings, and the overall architecture?"}
            ]
        },
        
        # Special characters in content
        {
            "name": "Special chars",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What does this symbol mean: @#$%^&*()!?"}
            ]
        },
        
        # Multiple languages
        {
            "name": "Multilingual",
            "messages": [
                {"role": "system", "content": "You are a multilingual assistant."},
                {"role": "user", "content": "Hello! 你好! Bonjour!"}
            ]
        },
        
        # Empty content (edge case)
        {
            "name": "Empty content",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": ""}
            ]
        },
        
        # Complex conversation
        {
            "name": "Complex conversation",
            "messages": [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "Write a Python function to calculate fibonacci numbers."},
                {"role": "assistant", "content": "Here's a Python function:\n\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"},
                {"role": "user", "content": "Can you optimize it?"},
                {"role": "assistant", "content": "Yes, here's an optimized version using memoization."},
                {"role": "user", "content": "Great! Now explain time complexity."}
            ]
        }
    ]
    
    print("=" * 60)
    print("PHASE 2.3: CHAT TEMPLATE APPLICATION")
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
    
    # Check if chat template is available
    pytorch_has_template = hasattr(pytorch_tokenizer, 'apply_chat_template')
    jax_has_template = hasattr(jax_tokenizer, 'apply_chat_template')
    
    print(f"PyTorch chat template support: {pytorch_has_template}")
    print(f"JAX chat template support: {jax_has_template}")
    
    if not (pytorch_has_template and jax_has_template):
        print("❌ Chat template not supported on both tokenizers")
        return False
    
    print()
    
    # Test each case
    all_match = True
    results = []
    
    print("Chat Template Comparison:")
    print("-" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        name = test_case["name"]
        messages = test_case["messages"]
        
        try:
            # Apply chat template with both tokenizers
            pytorch_text = pytorch_tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            jax_text = jax_tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Compare the resulting text
            text_match = pytorch_text == jax_text
            
            # Also tokenize the results for token comparison
            pytorch_tokens = pytorch_tokenizer.encode(pytorch_text)
            jax_tokens = jax_tokenizer.encode(jax_text)
            tokens_match = pytorch_tokens == jax_tokens
            
            # Overall match
            overall_match = text_match and tokens_match
            
            # Store result
            result = {
                'test_case': i,
                'name': name,
                'messages': messages,
                'pytorch_text': pytorch_text,
                'jax_text': jax_text,
                'pytorch_tokens': pytorch_tokens,
                'jax_tokens': jax_tokens,
                'text_match': text_match,
                'tokens_match': tokens_match,
                'overall_match': overall_match,
                'text_length': len(pytorch_text),
                'token_count': len(pytorch_tokens)
            }
            results.append(result)
            
            if not overall_match:
                all_match = False
            
            # Display result
            status = "✅" if overall_match else "❌"
            print(f"{status} Case {i:2d}: {name:<20} -> {len(pytorch_text)} chars, {len(pytorch_tokens)} tokens")
            
            # Show details for mismatches
            if not overall_match:
                if not text_match:
                    print(f"    Text differs:")
                    print(f"      PyTorch: {repr(pytorch_text[:100])}...")
                    print(f"      JAX:     {repr(jax_text[:100])}...")
                if not tokens_match:
                    print(f"    Tokens differ:")
                    print(f"      PyTorch: {pytorch_tokens[:20]}...")
                    print(f"      JAX:     {jax_tokens[:20]}...")
                    
        except Exception as e:
            print(f"❌ Case {i:2d}: {name:<20} -> ERROR: {e}")
            all_match = False
            result = {
                'test_case': i,
                'name': name,
                'messages': messages,
                'error': str(e),
                'overall_match': False
            }
            results.append(result)
    
    print("-" * 60)
    
    # Summary
    total_tests = len(test_cases)
    passed_tests = sum(1 for r in results if r.get('overall_match', False))
    error_tests = sum(1 for r in results if 'error' in r)
    
    print(f"\nSummary:")
    print(f"Total test cases: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Errors: {error_tests}")
    
    # Additional analysis for successful cases
    if passed_tests > 0:
        successful_results = [r for r in results if r.get('overall_match', False)]
        avg_text_length = sum(r['text_length'] for r in successful_results) / len(successful_results)
        avg_token_count = sum(r['token_count'] for r in successful_results) / len(successful_results)
        
        print(f"\nSuccessful Cases Analysis:")
        print(f"Average text length: {avg_text_length:.1f} characters")
        print(f"Average token count: {avg_token_count:.1f} tokens")
    
    if all_match:
        print("\n✅ ALL CHAT TEMPLATES MATCH!")
        print("Phase 2.3 PASSED - Ready for Phase 2.4")
    else:
        print("\n❌ CHAT TEMPLATE MISMATCHES DETECTED")
        print("Must fix chat template application before proceeding")
    
    return all_match

if __name__ == "__main__":
    success = test_chat_templates()
    sys.exit(0 if success else 1) 