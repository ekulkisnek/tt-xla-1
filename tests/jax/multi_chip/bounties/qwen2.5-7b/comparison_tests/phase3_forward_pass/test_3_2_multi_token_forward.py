#!/usr/bin/env python3
"""
Phase 3.2: Multi-Token Forward Pass Comparison
==============================================
Tests that both PyTorch and JAX models produce identical logits for multi-token sequences.
This helps identify if differences accumulate across sequence positions.
"""

import sys
import os
import numpy as np
import time
import gc
from pathlib import Path

def get_test_cases():
    """Get multi-token test cases of increasing complexity"""
    return [
        "Hello world",
        "The quick brown",
        "Python is great",
        "1 2 3 4 5",
        "Testing multi token sequences for comparison",
    ]

def run_pytorch_test(test_cases):
    """Run PyTorch model tests and save results"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print("Loading PyTorch model...")
        model_path = "../../weights"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model.eval()
        
        results = {}
        
        for test_input in test_cases:
            print(f"PyTorch processing: '{test_input}'")
            
            # Tokenize input
            inputs = tokenizer(test_input, return_tensors="pt")
            input_ids = inputs["input_ids"]
            
            # Forward pass
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits
            
            # Convert to float32 numpy for comparison
            logits_np = logits.float().numpy()
            tokens_np = input_ids.numpy()
            
            results[test_input] = {
                'logits': logits_np,
                'tokens': tokens_np,
                'shape': logits_np.shape,
                'seq_length': input_ids.shape[1]
            }
            
            print(f"  Shape: {logits_np.shape}")
            print(f"  Tokens: {tokens_np.tolist()}")
            print(f"  Sequence Length: {input_ids.shape[1]}")
        
        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return results, None
        
    except Exception as e:
        return None, str(e)

def run_jax_test(test_cases):
    """Run JAX model tests and save results"""
    try:
        import jax
        import jax.numpy as jnp
        import json
        sys.path.append("../..")
        
        # Import JAX model
        from qwen_jax_inference import Qwen25ForCausalLM, load_params
        from transformers import AutoTokenizer
        
        print("Loading JAX model...")
        model_path = "../../weights"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load config
        with open(f"{model_path}/config.json", 'r') as f:
            config = json.load(f)
        
        # Create model with bfloat16 to save memory
        model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
        
        # Load weights
        params = load_params(model, model_path, jnp.bfloat16)
        
        results = {}
        
        for test_input in test_cases:
            print(f"JAX processing: '{test_input}'")
            
            # Tokenize input
            inputs = tokenizer(test_input, return_tensors="np")
            input_ids = inputs["input_ids"]
            
            # Forward pass
            outputs = model.apply(
                params,
                input_ids=input_ids,
                return_dict=True
            )
            logits = outputs["logits"]
            
            # Convert to float32 numpy for comparison
            logits_np = np.array(logits, dtype=np.float32)
            
            results[test_input] = {
                'logits': logits_np,
                'tokens': input_ids,
                'shape': logits_np.shape,
                'seq_length': input_ids.shape[1]
            }
            
            print(f"  Shape: {logits_np.shape}")
            print(f"  Tokens: {input_ids.tolist()}")
            print(f"  Sequence Length: {input_ids.shape[1]}")
        
        # Cleanup
        del model, params
        jax.clear_caches()
        gc.collect()
        
        return results, None
        
    except Exception as e:
        return None, str(e)

def analyze_position_differences(pt_logits, jax_logits, test_input):
    """Analyze how differences vary across sequence positions"""
    
    seq_len = pt_logits.shape[1]
    position_analysis = []
    
    for pos in range(seq_len):
        pos_pt = pt_logits[0, pos, :]
        pos_jax = jax_logits[0, pos, :]
        
        abs_diff = np.abs(pos_pt - pos_jax)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        
        position_analysis.append({
            'position': pos,
            'max_diff': max_diff,
            'mean_diff': mean_diff
        })
        
        print(f"    Position {pos}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
    
    return position_analysis

def compare_results(pytorch_results, jax_results, tolerance=1e-4):
    """Compare results between PyTorch and JAX with position analysis"""
    
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    
    all_passed = True
    comparison_results = []
    position_analyses = {}
    
    for test_input in pytorch_results.keys():
        print(f"\nComparing: '{test_input}'")
        print("-" * 50)
        
        pt_data = pytorch_results[test_input]
        jax_data = jax_results[test_input]
        
        # Check tokens match
        if not np.array_equal(pt_data['tokens'], jax_data['tokens']):
            print(f"❌ Token mismatch!")
            comparison_results.append((test_input, False, "Token mismatch"))
            all_passed = False
            continue
        
        print(f"✅ Tokens match: {pt_data['tokens'].tolist()}")
        print(f"✅ Sequence length: {pt_data['seq_length']}")
        
        # Check shapes match
        if pt_data['shape'] != jax_data['shape']:
            print(f"❌ Shape mismatch!")
            comparison_results.append((test_input, False, "Shape mismatch"))
            all_passed = False
            continue
        
        print(f"✅ Shapes match: {pt_data['shape']}")
        
        # Overall logits comparison
        pt_logits = pt_data['logits']
        jax_logits = jax_data['logits']
        
        abs_diff = np.abs(pt_logits - jax_logits)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        
        print(f"\nOverall logits comparison:")
        print(f"  Max absolute difference: {max_diff:.2e}")
        print(f"  Mean absolute difference: {mean_diff:.2e}")
        print(f"  Tolerance threshold: {tolerance:.2e}")
        
        # Position-by-position analysis
        print(f"\nPosition-by-position analysis:")
        position_analysis = analyze_position_differences(pt_logits, jax_logits, test_input)
        position_analyses[test_input] = position_analysis
        
        if max_diff < tolerance:
            print(f"✅ LOGITS MATCH within tolerance")
            comparison_results.append((test_input, True, f"Max diff: {max_diff:.2e}"))
        else:
            print(f"❌ LOGITS DO NOT MATCH")
            comparison_results.append((test_input, False, f"Max diff: {max_diff:.2e}"))
            all_passed = False
    
    return all_passed, comparison_results, position_analyses

def summarize_position_trends(position_analyses):
    """Analyze trends across positions and inputs"""
    
    print("\n" + "=" * 70)
    print("POSITION TREND ANALYSIS")
    print("=" * 70)
    
    for test_input, analysis in position_analyses.items():
        print(f"\nSequence: '{test_input}'")
        print(f"Length: {len(analysis)} positions")
        
        max_diffs = [pos['max_diff'] for pos in analysis]
        mean_diffs = [pos['mean_diff'] for pos in analysis]
        
        print(f"Max differences across positions:")
        print(f"  Range: {min(max_diffs):.2e} to {max(max_diffs):.2e}")
        print(f"  Trend: {'Increasing' if max_diffs[-1] > max_diffs[0] else 'Decreasing' if max_diffs[-1] < max_diffs[0] else 'Stable'}")
        
        # Check if differences increase with position
        if len(analysis) > 1:
            correlation = np.corrcoef(range(len(analysis)), max_diffs)[0, 1]
            print(f"  Position correlation: {correlation:.3f} ({'positive' if correlation > 0.1 else 'negative' if correlation < -0.1 else 'neutral'})")

def main():
    """Main test function"""
    print("=" * 70)
    print("PHASE 3.2: MULTI-TOKEN FORWARD PASS COMPARISON")
    print("=" * 70)
    
    test_cases = get_test_cases()
    print(f"Test cases: {test_cases}")
    
    # Run PyTorch tests
    print(f"\n{'='*50}")
    print("RUNNING PYTORCH TESTS")
    print(f"{'='*50}")
    
    pytorch_results, pytorch_error = run_pytorch_test(test_cases)
    if pytorch_error:
        print(f"❌ PyTorch tests failed: {pytorch_error}")
        return False
    
    print("✅ PyTorch tests completed")
    
    # Run JAX tests
    print(f"\n{'='*50}")
    print("RUNNING JAX TESTS")
    print(f"{'='*50}")
    
    jax_results, jax_error = run_jax_test(test_cases)
    if jax_error:
        print(f"❌ JAX tests failed: {jax_error}")
        return False
    
    print("✅ JAX tests completed")
    
    # Compare results
    all_passed, comparison_results, position_analyses = compare_results(pytorch_results, jax_results)
    
    # Position trend analysis
    summarize_position_trends(position_analyses)
    
    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 3.2 FINAL SUMMARY")
    print("=" * 70)
    
    passed = 0
    for test_input, success, details in comparison_results:
        status = "✅ PASS" if success else "❌ FAIL"
        seq_len = pytorch_results[test_input]['seq_length']
        print(f"{test_input:<35} (len={seq_len:2d}) {status:<10} {details}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(comparison_results)} tests passed")
    
    if all_passed:
        print("🎉 ALL MULTI-TOKEN FORWARD PASS TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED - Investigating multi-token differences")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 