#!/usr/bin/env python3
"""
Phase 3.3: Embedding Analysis (Simplified)
==========================================
Compares embedding weights and outputs between PyTorch and JAX models.
This simplified version focuses on identifying if the issue starts at the embedding level.
"""

import sys
import os
import numpy as np
import time
import gc
import json
from pathlib import Path

def get_test_inputs():
    """Get test inputs for embedding analysis"""
    return [
        "Hello",   # Token 9707 from previous tests
        "The",     # Token 785 from previous tests  
        "world",   # Another common token
    ]

def extract_pytorch_embeddings(test_inputs):
    """Extract embedding weights and outputs from PyTorch model"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print("Loading PyTorch model for embedding analysis...")
        model_path = "../../weights"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model.eval()
        
        # Extract embedding weights
        embedding_weights = model.model.embed_tokens.weight.data
        print(f"PyTorch embedding weights shape: {embedding_weights.shape}")
        
        results = {}
        
        for test_input in test_inputs:
            # Tokenize
            inputs = tokenizer(test_input, return_tensors="pt")
            input_ids = inputs["input_ids"]
            token_id = input_ids[0, 0].item()
            
            # Get embedding output
            with torch.no_grad():
                embedding_output = model.model.embed_tokens(input_ids)
            
            # Store results
            results[test_input] = {
                'token_id': token_id,
                'embedding_weight': embedding_weights[token_id].float().numpy(),
                'embedding_output': embedding_output.float().numpy(),
                'weight_shape': embedding_weights[token_id].shape,
                'output_shape': embedding_output.shape
            }
            
            print(f"PyTorch '{test_input}' -> token {token_id}")
            print(f"  Weight shape: {embedding_weights[token_id].shape}")
            print(f"  Output shape: {embedding_output.shape}")
        
        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return results, None
        
    except Exception as e:
        return None, str(e)

def extract_jax_embeddings(test_inputs):
    """Extract embedding weights and outputs from JAX model"""
    try:
        import jax
        import jax.numpy as jnp
        import json
        sys.path.append("../..")
        
        from qwen_jax_inference import Qwen25ForCausalLM, load_params
        from transformers import AutoTokenizer
        
        print("Loading JAX model for embedding analysis...")
        model_path = "../../weights"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load config and create model
        with open(f"{model_path}/config.json", 'r') as f:
            config = json.load(f)
        
        model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
        params = load_params(model, model_path, jnp.bfloat16)
        
        # Extract embedding weights from parameters
        embedding_weights = params['params']['embed_tokens']['embedding']
        print(f"JAX embedding weights shape: {embedding_weights.shape}")
        
        results = {}
        
        for test_input in test_inputs:
            # Tokenize
            inputs = tokenizer(test_input, return_tensors="np")
            input_ids = inputs["input_ids"]
            token_id = input_ids[0, 0].item()
            
            # Get embedding weight for this token
            embedding_weight = embedding_weights[token_id]
            
            # Manual embedding lookup (simulating model.embed_tokens)
            embedding_output = embedding_weight[None, None, :]  # [1, 1, hidden_size]
            
            results[test_input] = {
                'token_id': token_id,
                'embedding_weight': np.array(embedding_weight, dtype=np.float32),
                'embedding_output': np.array(embedding_output, dtype=np.float32),
                'weight_shape': embedding_weight.shape,
                'output_shape': embedding_output.shape
            }
            
            print(f"JAX '{test_input}' -> token {token_id}")
            print(f"  Weight shape: {embedding_weight.shape}")
            print(f"  Output shape: {embedding_output.shape}")
        
        # Cleanup
        del model, params
        jax.clear_caches()
        gc.collect()
        
        return results, None
        
    except Exception as e:
        return None, str(e)

def compare_embeddings(pytorch_results, jax_results, tolerance=1e-6):
    """Compare embedding weights and outputs between PyTorch and JAX"""
    
    print("\n" + "=" * 70)
    print("EMBEDDING WEIGHT & OUTPUT COMPARISON")
    print("=" * 70)
    
    comparison_results = []
    
    for test_input in pytorch_results.keys():
        print(f"\nAnalyzing: '{test_input}'")
        print("-" * 40)
        
        pt_data = pytorch_results[test_input]
        jax_data = jax_results[test_input]
        
        # Check token IDs match
        if pt_data['token_id'] != jax_data['token_id']:
            print(f"❌ Token ID mismatch: PyTorch {pt_data['token_id']} vs JAX {jax_data['token_id']}")
            comparison_results.append((test_input, False, "Token ID mismatch"))
            continue
        
        print(f"✅ Token ID matches: {pt_data['token_id']}")
        
        # Compare embedding weights
        pt_weight = pt_data['embedding_weight']
        jax_weight = jax_data['embedding_weight']
        
        weight_diff = np.abs(pt_weight - jax_weight)
        max_weight_diff = np.max(weight_diff)
        mean_weight_diff = np.mean(weight_diff)
        
        print(f"\nEmbedding Weight Comparison:")
        print(f"  Max weight difference: {max_weight_diff:.2e}")
        print(f"  Mean weight difference: {mean_weight_diff:.2e}")
        
        # Compare embedding outputs
        pt_output = pt_data['embedding_output']
        jax_output = jax_data['embedding_output']
        
        output_diff = np.abs(pt_output - jax_output)
        max_output_diff = np.max(output_diff)
        mean_output_diff = np.mean(output_diff)
        
        print(f"\nEmbedding Output Comparison:")
        print(f"  Max output difference: {max_output_diff:.2e}")
        print(f"  Mean output difference: {mean_output_diff:.2e}")
        print(f"  Tolerance threshold: {tolerance:.2e}")
        
        # Determine if weights and outputs match
        weights_match = max_weight_diff < tolerance
        outputs_match = max_output_diff < tolerance
        
        if weights_match and outputs_match:
            print(f"✅ EMBEDDINGS MATCH")
            status = "MATCH"
            overall_diff = max(max_weight_diff, max_output_diff)
        else:
            print(f"❌ EMBEDDINGS DIFFER")
            if not weights_match:
                print(f"  Weight difference: {max_weight_diff:.2e} > {tolerance:.2e}")
            if not outputs_match:
                print(f"  Output difference: {max_output_diff:.2e} > {tolerance:.2e}")
            status = "DIFFER"
            overall_diff = max(max_weight_diff, max_output_diff)
        
        comparison_results.append((test_input, weights_match and outputs_match, status, overall_diff))
    
    return comparison_results

def main():
    """Main test function"""
    print("=" * 70)
    print("PHASE 3.3: EMBEDDING ANALYSIS (SIMPLIFIED)")
    print("=" * 70)
    
    test_inputs = get_test_inputs()
    print(f"Test inputs: {test_inputs}")
    
    # Extract PyTorch embeddings
    print(f"\n{'='*50}")
    print("EXTRACTING PYTORCH EMBEDDINGS")
    print(f"{'='*50}")
    
    pytorch_results, pytorch_error = extract_pytorch_embeddings(test_inputs)
    if pytorch_error:
        print(f"❌ PyTorch extraction failed: {pytorch_error}")
        return False
    
    print("✅ PyTorch embeddings extracted")
    
    # Extract JAX embeddings
    print(f"\n{'='*50}")
    print("EXTRACTING JAX EMBEDDINGS")
    print(f"{'='*50}")
    
    jax_results, jax_error = extract_jax_embeddings(test_inputs)
    if jax_error:
        print(f"❌ JAX extraction failed: {jax_error}")
        return False
    
    print("✅ JAX embeddings extracted")
    
    # Compare embeddings
    comparison_results = compare_embeddings(pytorch_results, jax_results)
    
    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 3.3 FINAL SUMMARY")
    print("=" * 70)
    
    matching_embeddings = sum(1 for _, matches, _, _ in comparison_results if matches)
    total_embeddings = len(comparison_results)
    
    print(f"Embeddings analyzed: {total_embeddings}")
    print(f"Matching embeddings: {matching_embeddings}")
    print(f"Differing embeddings: {total_embeddings - matching_embeddings}")
    
    for test_input, matches, status, diff in comparison_results:
        status_symbol = "✅" if matches else "❌"
        print(f"  '{test_input}':<10 {status_symbol} {status:<8} max_diff: {diff:.2e}")
    
    if matching_embeddings == total_embeddings:
        print("\n✅ ALL EMBEDDINGS MATCH!")
        print("The issue does NOT start at the embedding level.")
        print("Focus investigation on transformer layers and attention mechanism.")
        return True
    else:
        print("\n❌ EMBEDDING DIFFERENCES DETECTED!")
        print("The issue starts at the embedding level - check weight loading.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 