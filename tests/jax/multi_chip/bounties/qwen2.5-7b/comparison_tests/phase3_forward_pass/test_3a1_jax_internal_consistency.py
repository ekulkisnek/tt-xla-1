#!/usr/bin/env python3
"""
Phase 3A.1: JAX Model Internal Consistency Check
===============================================
Test the JAX model using its own components rather than manual computation.
This will verify if the JAX model itself computes correctly when using actual model layers.
"""

import sys
import os
import numpy as np
import time
import gc
import json
import math
from pathlib import Path

def get_test_input():
    """Get test input for internal consistency check"""
    return "Hello"  # Token 9707 - confirmed identical embedding

def extract_pytorch_reference(test_input):
    """Extract PyTorch reference computation using actual model layers"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print("Loading PyTorch model for reference...")
        model_path = "../../weights"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model.eval()
        
        # Tokenize input
        inputs = tokenizer(test_input, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        print(f"PyTorch input: {test_input}")
        print(f"PyTorch tokens: {input_ids.tolist()}")
        
        # Get Layer 0 computation using actual PyTorch components
        with torch.no_grad():
            # Step 1: Embedding
            embedding_output = model.model.embed_tokens(input_ids)
            
            # Step 2: Layer 0 input normalization
            layer_0 = model.model.layers[0]
            normalized_input = layer_0.input_layernorm(embedding_output)
            
            # Step 3: Layer 0 attention (using actual layer, not manual computation)
            attn_output = layer_0.self_attn(
                normalized_input,
                attention_mask=None,
                position_ids=None,
                past_key_value=None
            )[0]  # Get just the output, not the attention weights
            
            # Step 4: Residual connection after attention
            hidden_after_attn = embedding_output + attn_output
            
            # Step 5: Post-attention normalization
            normalized_after_attn = layer_0.post_attention_layernorm(hidden_after_attn)
            
            # Step 6: MLP using actual layer
            mlp_output = layer_0.mlp(normalized_after_attn)
            
            # Step 7: Final residual connection (Layer 0 complete)
            layer_0_final = hidden_after_attn + mlp_output
            
            # Step 8: Full forward pass for comparison
            full_outputs = model(input_ids)
            final_logits = full_outputs.logits
            
        results = {
            'embedding_output': embedding_output.float().numpy(),
            'normalized_input': normalized_input.float().numpy(),
            'attn_output': attn_output.float().numpy(),
            'hidden_after_attn': hidden_after_attn.float().numpy(),
            'normalized_after_attn': normalized_after_attn.float().numpy(),
            'mlp_output': mlp_output.float().numpy(),
            'layer_0_final': layer_0_final.float().numpy(),
            'final_logits': final_logits.float().numpy(),
            'tokens': input_ids.numpy()
        }
        
        print(f"PyTorch reference components extracted:")
        for key, value in results.items():
            if key != 'tokens':
                print(f"  {key}: shape {value.shape}")
        
        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return results, None
        
    except Exception as e:
        return None, str(e)

def extract_jax_actual_components(test_input):
    """Extract JAX computation using actual model components (not manual)"""
    try:
        import jax
        import jax.numpy as jnp
        sys.path.append("../..")
        
        from qwen_jax_inference import Qwen25ForCausalLM, load_params
        from transformers import AutoTokenizer
        
        print("Loading JAX model for actual component testing...")
        model_path = "../../weights"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load config and model
        with open(f"{model_path}/config.json", 'r') as f:
            config = json.load(f)
        
        model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
        params = load_params(model, model_path, jnp.bfloat16)
        
        # Tokenize input
        inputs = tokenizer(test_input, return_tensors="np")
        input_ids = jnp.array(inputs["input_ids"])
        
        print(f"JAX input: {test_input}")
        print(f"JAX tokens: {input_ids.tolist()}")
        
        # Use actual JAX model components via proper Flax apply patterns
        batch_size, seq_len = input_ids.shape
        
        # Step 1: Embedding - access via model apply
        embedding_output = model.apply(
            {'params': params['params']}, 
            input_ids,
            method=lambda module, input_ids: module.embed_tokens(input_ids)
        )
        
        # Step 2: Layer 0 components using model apply with method access
        # Step 2a: Input normalization
        normalized_input = model.apply(
            {'params': params['params']}, 
            embedding_output,
            method=lambda module, x: module.layers[0].input_layernorm(x)
        )
        
        # Step 2b: Attention using actual layer
        # Create position_ids for attention
        position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
        
        attn_output, _ = model.apply(
            {'params': params['params']},
            normalized_input,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=None,
            method=lambda module, hidden_states, attention_mask, position_ids, past_key_value: 
                   module.layers[0].self_attn(hidden_states, attention_mask, position_ids, past_key_value)
        )
        
        # Step 3: Residual connection after attention
        hidden_after_attn = embedding_output + attn_output
        
        # Step 4: Post-attention normalization
        normalized_after_attn = model.apply(
            {'params': params['params']},
            hidden_after_attn,
            method=lambda module, x: module.layers[0].post_attention_layernorm(x)
        )
        
        # Step 5: MLP
        mlp_output = model.apply(
            {'params': params['params']},
            normalized_after_attn,
            method=lambda module, x: module.layers[0].mlp(x)
        )
        
        # Step 6: Final residual connection (Layer 0 complete)
        layer_0_final = hidden_after_attn + mlp_output
        
        # Step 7: Full forward pass using actual model for comparison
        full_outputs = model.apply(
            {'params': params['params']},
            input_ids
        )
        final_logits = full_outputs['logits']
        
        results = {
            'embedding_output': np.array(embedding_output.astype(jnp.float32)),
            'normalized_input': np.array(normalized_input.astype(jnp.float32)),
            'attn_output': np.array(attn_output.astype(jnp.float32)),
            'hidden_after_attn': np.array(hidden_after_attn.astype(jnp.float32)),
            'normalized_after_attn': np.array(normalized_after_attn.astype(jnp.float32)),
            'mlp_output': np.array(mlp_output.astype(jnp.float32)),
            'layer_0_final': np.array(layer_0_final.astype(jnp.float32)),
            'final_logits': np.array(final_logits.astype(jnp.float32)),
            'tokens': np.array(input_ids)
        }
        
        print(f"JAX actual components extracted:")
        for key, value in results.items():
            if key != 'tokens':
                print(f"  {key}: shape {value.shape}")
        
        # Cleanup
        del model, params
        jax.clear_caches()
        gc.collect()
        
        return results, None
        
    except Exception as e:
        import traceback
        return None, f"{str(e)}\n{traceback.format_exc()}"

def compare_pytorch_jax_actual(pytorch_results, jax_results):
    """Compare PyTorch vs JAX using actual model components"""
    print("\n" + "=" * 70)
    print("PYTORCH vs JAX ACTUAL MODEL COMPONENT COMPARISON")
    print("=" * 70)
    
    tolerance = 1e-6
    large_tolerance = 1e-4  # For initial assessment
    
    all_passed = True
    results_summary = {}
    
    # Components to compare
    components = [
        'embedding_output',
        'normalized_input', 
        'attn_output',
        'hidden_after_attn',
        'normalized_after_attn',
        'mlp_output',
        'layer_0_final',
        'final_logits'
    ]
    
    for component in components:
        if component not in pytorch_results or component not in jax_results:
            print(f"⚠️ {component} missing in one implementation")
            continue
            
        pt_data = pytorch_results[component]
        jax_data = jax_results[component]
        
        print(f"\n--- {component.upper()} ---")
        print(f"PyTorch shape: {pt_data.shape}")
        print(f"JAX shape: {jax_data.shape}")
        
        if pt_data.shape != jax_data.shape:
            print(f"❌ Shape mismatch!")
            all_passed = False
            continue
        
        diff = np.abs(pt_data - jax_data)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"Max difference: {max_diff:.8e}")
        print(f"Mean difference: {mean_diff:.8e}")
        
        # Check against strict tolerance
        if max_diff < tolerance:
            print(f"✅ {component} PERFECT MATCH (< 1e-6)")
            status = "PERFECT"
        elif max_diff < large_tolerance:
            print(f"🟡 {component} CLOSE (< 1e-4)")
            status = "CLOSE"
            all_passed = False
        else:
            print(f"❌ {component} DIFFERS (> 1e-4)")
            status = "DIFFERS"
            all_passed = False
            
            # Show sample differences for failed cases
            print(f"Sample PyTorch: {pt_data.flat[:5]}")
            print(f"Sample JAX: {jax_data.flat[:5]}")
            print(f"Sample diff: {diff.flat[:5]}")
        
        results_summary[component] = {
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'status': status
        }
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - ACTUAL MODEL COMPONENTS")
    print("=" * 70)
    
    perfect_count = sum(1 for r in results_summary.values() if r['status'] == 'PERFECT')
    close_count = sum(1 for r in results_summary.values() if r['status'] == 'CLOSE')
    differs_count = sum(1 for r in results_summary.values() if r['status'] == 'DIFFERS')
    
    print(f"Perfect matches (< 1e-6): {perfect_count}/{len(results_summary)}")
    print(f"Close matches (< 1e-4): {close_count}/{len(results_summary)}")
    print(f"Significant differences: {differs_count}/{len(results_summary)}")
    
    if all_passed:
        print("🎉 ALL COMPONENTS MATCH WITHIN 1e-6 TOLERANCE!")
        print("✅ Phase 3A.1 PASSED - JAX model internal consistency verified")
    else:
        print("❌ Some components still differ")
        print("🔍 Need to investigate specific component differences")
    
    return all_passed, results_summary

def main():
    """Run Phase 3A.1: JAX Model Internal Consistency Check"""
    print("=" * 70)
    print("PHASE 3A.1: JAX MODEL INTERNAL CONSISTENCY CHECK")
    print("=" * 70)
    print("Testing JAX model using actual model components (not manual computation)")
    
    test_input = get_test_input()
    
    # Extract PyTorch reference
    print(f"\n🔄 Extracting PyTorch reference...")
    pytorch_results, pt_error = extract_pytorch_reference(test_input)
    if pytorch_results is None:
        print(f"❌ PyTorch extraction failed: {pt_error}")
        return False
    
    # Extract JAX actual components
    print(f"\n🔄 Extracting JAX actual components...")
    jax_results, jax_error = extract_jax_actual_components(test_input)
    if jax_results is None:
        print(f"❌ JAX extraction failed: {jax_error}")
        return False
    
    # Compare results
    print(f"\n🔄 Comparing actual model components...")
    success, summary = compare_pytorch_jax_actual(pytorch_results, jax_results)
    
    # Save results
    timestamp = int(time.time())
    results = {
        'test_input': test_input,
        'pytorch_results': {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in pytorch_results.items()},
        'jax_results': {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in jax_results.items()},
        'comparison_summary': summary,
        'overall_success': success,
        'timestamp': timestamp
    }
    
    output_file = f"RESULTS_3A1_internal_consistency_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: {output_file}")
    
    if success:
        print("\n🎉 PHASE 3A.1 COMPLETE: JAX model internal consistency verified!")
        print("✅ Ready to proceed to Phase 3A.3 and Phase 3B.1a")
    else:
        print("\n🔍 PHASE 3A.1 INCOMPLETE: Some components still differ")
        print("📋 Next: Investigate specific component differences")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 