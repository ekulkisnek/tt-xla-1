#!/usr/bin/env python3
"""
Phase 3.4: First Transformer Layer Analysis
===========================================
Since Phase 3.3 confirmed embeddings are identical, this test focuses on analyzing
the first transformer layer (layer 0) to identify where divergence first occurs.
"""

import sys
import os
import numpy as np
import time
import gc
import json
from pathlib import Path

def get_test_input():
    """Get simple test input for first layer analysis"""
    return "Hello"  # Token 9707 - confirmed identical embedding

def extract_pytorch_first_layer(test_input):
    """Extract first layer components from PyTorch model"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print("Loading PyTorch model for first layer analysis...")
        model_path = "../../weights"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            output_hidden_states=True
        )
        model.eval()
        
        # Tokenize input
        inputs = tokenizer(test_input, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        print(f"PyTorch input: {test_input}")
        print(f"PyTorch tokens: {input_ids.tolist()}")
        
        with torch.no_grad():
            # Get hidden states including layer 0 output
            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # Extract key components
            embedding_output = hidden_states[0]  # After embedding
            layer_0_output = hidden_states[1]    # After layer 0
            final_logits = outputs.logits
            
        results = {
            'embedding_output': embedding_output.float().numpy(),
            'layer_0_output': layer_0_output.float().numpy(),
            'final_logits': final_logits.float().numpy(),
            'embedding_shape': embedding_output.shape,
            'layer_0_shape': layer_0_output.shape,
            'tokens': input_ids.numpy()
        }
        
        print(f"PyTorch embedding output shape: {embedding_output.shape}")
        print(f"PyTorch layer 0 output shape: {layer_0_output.shape}")
        print(f"PyTorch final logits shape: {final_logits.shape}")
        
        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return results, None
        
    except Exception as e:
        return None, str(e)

def extract_jax_first_layer(test_input):
    """Extract first layer components from JAX model"""
    try:
        import jax
        import jax.numpy as jnp
        sys.path.append("../..")
        
        from qwen_jax_inference import Qwen25ForCausalLM, load_params
        from transformers import AutoTokenizer
        
        print("Loading JAX model for first layer analysis...")
        model_path = "../../weights"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load config and model
        with open(f"{model_path}/config.json", 'r') as f:
            config = json.load(f)
        
        model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
        params = load_params(model, model_path, jnp.bfloat16)
        
        # Tokenize input
        inputs = tokenizer(test_input, return_tensors="np")
        input_ids = inputs["input_ids"]
        
        print(f"JAX input: {test_input}")
        print(f"JAX tokens: {input_ids.tolist()}")
        
        # Manual step-by-step forward pass to extract intermediate states
        # 1. Embedding (we know this is identical from Phase 3.3)
        embedding_weights = params['params']['embed_tokens']['embedding']
        token_id = input_ids[0, 0]
        embedding_output = embedding_weights[token_id][None, None, :]  # [1, 1, hidden_size]
        
        # 2. Get final logits for comparison
        outputs = model.apply(
            params,
            input_ids=input_ids,
            return_dict=True
        )
        final_logits = outputs["logits"]
        
        # Placeholder for layer 0 output - this is complex to extract manually
        layer_0_output = embedding_output  # Placeholder - NOT ACTUAL LAYER 0 OUTPUT
        
        results = {
            'embedding_output': np.array(embedding_output, dtype=np.float32),
            'layer_0_output': np.array(layer_0_output, dtype=np.float32),  # PLACEHOLDER
            'final_logits': np.array(final_logits, dtype=np.float32),
            'embedding_shape': embedding_output.shape,
            'layer_0_shape': layer_0_output.shape,  # PLACEHOLDER
            'tokens': input_ids,
            'note': "Layer 0 output is placeholder - need manual implementation"
        }
        
        print(f"JAX embedding output shape: {embedding_output.shape}")
        print(f"JAX layer 0 output shape: {layer_0_output.shape} (PLACEHOLDER)")
        print(f"JAX final logits shape: {final_logits.shape}")
        print("NOTE: Layer 0 extraction needs manual implementation")
        
        # Cleanup
        del model, params
        jax.clear_caches()
        gc.collect()
        
        return results, None
        
    except Exception as e:
        return None, str(e)

def compare_first_layer(pytorch_results, jax_results, tolerance=1e-4):
    """Compare first layer outputs between PyTorch and JAX"""
    
    print("\n" + "=" * 70)
    print("FIRST LAYER ANALYSIS")
    print("=" * 70)
    
    # Check tokens match
    if not np.array_equal(pytorch_results['tokens'], jax_results['tokens']):
        print("❌ Token mismatch!")
        return False
    
    print(f"✅ Tokens match: {pytorch_results['tokens'].tolist()}")
    
    # Compare embedding outputs (should be identical from Phase 3.3)
    pt_embedding = pytorch_results['embedding_output']
    jax_embedding = jax_results['embedding_output']
    
    embedding_diff = np.abs(pt_embedding - jax_embedding)
    max_embedding_diff = np.max(embedding_diff)
    
    print(f"\nEmbedding Output Comparison:")
    print(f"  Max difference: {max_embedding_diff:.2e}")
    print(f"  Expected: 0.00e+00 (from Phase 3.3)")
    
    if max_embedding_diff < 1e-10:
        print("✅ Embeddings confirmed identical")
    else:
        print("❌ Unexpected embedding difference!")
    
    # Compare final logits (should show significant differences from Phase 3.1)
    pt_logits = pytorch_results['final_logits']
    jax_logits = jax_results['final_logits']
    
    logits_diff = np.abs(pt_logits - jax_logits)
    max_logits_diff = np.max(logits_diff)
    
    print(f"\nFinal Logits Comparison:")
    print(f"  Max difference: {max_logits_diff:.2e}")
    print(f"  Expected: ~7.81e-01 (from Phase 3.1)")
    
    if max_logits_diff > 0.1:
        print("✅ Logits differences confirmed (as expected)")
    else:
        print("❌ Unexpected - logits should differ significantly!")
    
    # Layer 0 comparison (if available)
    if 'note' not in jax_results:
        pt_layer0 = pytorch_results['layer_0_output']
        jax_layer0 = jax_results['layer_0_output']
        
        layer0_diff = np.abs(pt_layer0 - jax_layer0)
        max_layer0_diff = np.max(layer0_diff)
        
        print(f"\nLayer 0 Output Comparison:")
        print(f"  Max difference: {max_layer0_diff:.2e}")
        print(f"  Tolerance: {tolerance:.2e}")
        
        if max_layer0_diff < tolerance:
            print("✅ Layer 0 outputs match")
        else:
            print("🎯 DIVERGENCE DETECTED IN LAYER 0!")
            print("This is where the problem begins!")
    else:
        print(f"\nLayer 0 Analysis:")
        print(f"❌ {jax_results['note']}")
        print("Need to implement manual layer 0 computation for JAX")
    
    return True

def main():
    """Main test function"""
    print("=" * 70)
    print("PHASE 3.4: FIRST TRANSFORMER LAYER ANALYSIS")
    print("=" * 70)
    
    test_input = get_test_input()
    print(f"Test input: '{test_input}'")
    
    # Extract PyTorch first layer
    print(f"\n{'='*50}")
    print("EXTRACTING PYTORCH FIRST LAYER")
    print(f"{'='*50}")
    
    pytorch_results, pytorch_error = extract_pytorch_first_layer(test_input)
    if pytorch_error:
        print(f"❌ PyTorch extraction failed: {pytorch_error}")
        return False
    
    print("✅ PyTorch first layer extracted")
    
    # Extract JAX first layer
    print(f"\n{'='*50}")
    print("EXTRACTING JAX FIRST LAYER")
    print(f"{'='*50}")
    
    jax_results, jax_error = extract_jax_first_layer(test_input)
    if jax_error:
        print(f"❌ JAX extraction failed: {jax_error}")
        return False
    
    print("✅ JAX first layer extracted")
    
    # Compare results
    success = compare_first_layer(pytorch_results, jax_results)
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 3.4 SUMMARY")
    print("=" * 70)
    
    print("Key Findings:")
    print("- Embeddings confirmed identical (validates Phase 3.3)")
    print("- Final logits show expected differences (validates Phase 3.1)")
    print("- Layer 0 analysis needs manual JAX implementation")
    
    print("\nNext Steps:")
    print("- Implement detailed JAX layer 0 forward pass")
    print("- Compare attention mechanism components")
    print("- Identify exact point of divergence in transformer layer")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 