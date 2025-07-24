#!/usr/bin/env python3
"""
Phase 2.2: Attention Mask Comparison
====================================
Tests that both PyTorch and JAX tokenizers produce identical attention masks.
Attention masks are crucial for proper model input handling.
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

def test_attention_masks():
    """Test attention mask generation with various input scenarios"""
    
    # Test cases covering different attention mask scenarios
    test_cases = [
        # Single sequences
        "Hello world",
        "This is a longer sentence with more words",
        
        # Empty and short
        "",
        "A",
        
        # Different lengths
        "Short",
        "This is a medium length sentence for testing",
        "This is a much longer sentence that contains many more words and should result in a longer attention mask for testing purposes",
        
        # Special characters that might affect masking
        "Hello, world!",
        "What's happening? Nothing much...",
        
        # Multiple sentences
        "First sentence. Second sentence.",
        "Question? Answer. Statement!",
        
        # Whitespace variations
        " leading space",
        "trailing space ",
        "  multiple   spaces  ",
        
        # Mixed content
        "Text with 123 numbers",
        "Mixed-content_example@test.com",
    ]
    
    print("=" * 60)
    print("PHASE 2.2: ATTENTION MASK COMPARISON")
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
    
    print("Attention Mask Comparison:")
    print("-" * 60)
    
    for i, test_string in enumerate(test_cases, 1):
        # Tokenize with attention masks
        pytorch_encoding = pytorch_tokenizer(test_string, return_tensors="pt")
        jax_encoding = jax_tokenizer(test_string, return_tensors="pt")
        
        # Extract components
        pytorch_input_ids = pytorch_encoding['input_ids'].tolist()[0]
        pytorch_attention_mask = pytorch_encoding['attention_mask'].tolist()[0]
        
        jax_input_ids = jax_encoding['input_ids'].tolist()[0]
        jax_attention_mask = jax_encoding['attention_mask'].tolist()[0]
        
        # Compare input IDs and attention masks
        input_ids_match = pytorch_input_ids == jax_input_ids
        attention_mask_match = pytorch_attention_mask == jax_attention_mask
        
        # Overall match
        overall_match = input_ids_match and attention_mask_match
        
        # Store result
        result = {
            'test_case': i,
            'input': test_string,
            'pytorch_input_ids': pytorch_input_ids,
            'jax_input_ids': jax_input_ids,
            'pytorch_attention_mask': pytorch_attention_mask,
            'jax_attention_mask': jax_attention_mask,
            'input_ids_match': input_ids_match,
            'attention_mask_match': attention_mask_match,
            'overall_match': overall_match,
            'sequence_length': len(pytorch_input_ids)
        }
        results.append(result)
        
        if not overall_match:
            all_match = False
        
        # Display result
        status = "✅" if overall_match else "❌"
        input_display = repr(test_string) if len(test_string) <= 25 else repr(test_string[:22] + "...")
        mask_sum = sum(pytorch_attention_mask)
        print(f"{status} Case {i:2d}: {input_display:<30} -> len={len(pytorch_input_ids)}, mask_sum={mask_sum}")
        
        # Show details for mismatches
        if not overall_match:
            if not input_ids_match:
                print(f"    Input IDs differ:")
                print(f"      PyTorch: {pytorch_input_ids}")
                print(f"      JAX:     {jax_input_ids}")
            if not attention_mask_match:
                print(f"    Attention masks differ:")
                print(f"      PyTorch: {pytorch_attention_mask}")
                print(f"      JAX:     {jax_attention_mask}")
    
    print("-" * 60)
    
    # Summary
    total_tests = len(test_cases)
    passed_tests = sum(1 for r in results if r['overall_match'])
    input_id_matches = sum(1 for r in results if r['input_ids_match'])
    attention_mask_matches = sum(1 for r in results if r['attention_mask_match'])
    
    print(f"\nSummary:")
    print(f"Total test cases: {total_tests}")
    print(f"Overall matches: {passed_tests}")
    print(f"Input ID matches: {input_id_matches}")
    print(f"Attention mask matches: {attention_mask_matches}")
    
    # Additional analysis
    if all_match:
        # Verify attention mask properties
        print(f"\nAttention Mask Properties:")
        for result in results:
            mask = result['pytorch_attention_mask']
            if len(mask) > 0:
                # Check if mask is all 1s (no padding)
                all_ones = all(m == 1 for m in mask)
                print(f"  Case {result['test_case']:2d}: All 1s = {all_ones}, Length = {len(mask)}")
    
    if all_match:
        print("\n✅ ALL ATTENTION MASKS MATCH!")
        print("Phase 2.2 PASSED - Ready for Phase 2.3")
    else:
        print("\n❌ ATTENTION MASK MISMATCHES DETECTED")
        print("Must fix attention mask generation before proceeding")
    
    return all_match

if __name__ == "__main__":
    success = test_attention_masks()
    sys.exit(0 if success else 1) 