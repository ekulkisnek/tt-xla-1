#!/usr/bin/env python3
"""
Phase 3.3: Layer-by-Layer Intermediate Analysis
===============================================
Extracts and compares hidden states after each transformer layer to identify
exactly where the divergence between PyTorch and JAX models first occurs.
"""

import sys
import os
import numpy as np
import time
import gc
import json
from pathlib import Path

def get_test_input():
    """Get a simple test input for layer analysis"""
    # Use simple input from Phase 3.1 that showed clear differences
    return "Hello"

def extract_pytorch_layer_states(test_input):
    """Extract hidden states after each layer from PyTorch model"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print("Loading PyTorch model for layer extraction...")
        model_path = "../../weights"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            output_hidden_states=True,  # Enable hidden state extraction
        )
        model.eval()
        
        # Tokenize input
        inputs = tokenizer(test_input, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        print(f"PyTorch input: {test_input}")
        print(f"PyTorch tokens: {input_ids.tolist()}")
        print(f"PyTorch input shape: {input_ids.shape}")
        
        # Forward pass with hidden states
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of (num_layers + 1) tensors
            final_logits = outputs.logits
        
        # Convert to numpy and collect layer states
        layer_states = {}
        
        # Hidden states includes: [embedding_output, layer_0_output, layer_1_output, ..., layer_27_output]
        print(f"PyTorch extracted {len(hidden_states)} hidden state tensors")
        
        for i, layer_hidden in enumerate(hidden_states):
            layer_name = "embedding" if i == 0 else f"layer_{i-1}"
            layer_states[layer_name] = {
                'hidden_state': layer_hidden.float().numpy(),
                'shape': layer_hidden.shape,
                'layer_index': i
            }
            print(f"  {layer_name}: shape {layer_hidden.shape}")
        
        # Add final logits
        layer_states['final_logits'] = {
            'hidden_state': final_logits.float().numpy(),
            'shape': final_logits.shape,
            'layer_index': len(hidden_states)
        }
        
        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return layer_states, input_ids.numpy(), None
        
    except Exception as e:
        return None, None, str(e)

def extract_jax_layer_states(test_input):
    """Extract hidden states after each layer from JAX model"""
    try:
        import jax
        import jax.numpy as jnp
        import json
        sys.path.append("../..")
        
        # Import JAX model components
        from qwen_jax_inference import Qwen25ForCausalLM, load_params, QwenDecoderLayer
        from transformers import AutoTokenizer
        
        print("Loading JAX model for layer extraction...")
        model_path = "../../weights"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load config
        with open(f"{model_path}/config.json", 'r') as f:
            config = json.load(f)
        
        # Create model with bfloat16
        model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
        
        # Load weights
        params = load_params(model, model_path, jnp.bfloat16)
        
        # Tokenize input
        inputs = tokenizer(test_input, return_tensors="np")
        input_ids = inputs["input_ids"]
        
        print(f"JAX input: {test_input}")
        print(f"JAX tokens: {input_ids.tolist()}")
        print(f"JAX input shape: {input_ids.shape}")
        
        # Manual layer-by-layer forward pass
        layer_states = {}
        batch, seq = input_ids.shape
        
        # Start with embedding
        hidden_states = model.apply(
            {'params': params['params']['embed_tokens']},
            input_ids,
            method=lambda module, input_ids: module.embed_tokens(input_ids)
        )
        
        layer_states['embedding'] = {
            'hidden_state': np.array(hidden_states, dtype=np.float32),
            'shape': hidden_states.shape,
            'layer_index': 0
        }
        
        print(f"JAX embedding shape: {hidden_states.shape}")
        
        # Process each transformer layer
        current_hidden = hidden_states
        
        for layer_idx in range(config["num_hidden_layers"]):
            layer_name = f"layer_{layer_idx}"
            print(f"  Processing {layer_name}...")
            
            # Apply single decoder layer
            layer_params = {'params': params['params'][f'layers_{layer_idx}']}
            
            # Create a temporary layer model for this specific layer
            temp_layer = QwenDecoderLayer(config=config, dtype=jnp.bfloat16)
            
            # Apply the layer
            current_hidden, _ = temp_layer.apply(
                layer_params,
                current_hidden,
                attention_mask=None,
                position_ids=None,
                past_key_value=None
            )
            
            layer_states[layer_name] = {
                'hidden_state': np.array(current_hidden, dtype=np.float32),
                'shape': current_hidden.shape,
                'layer_index': layer_idx + 1
            }
            
            print(f"    {layer_name} output shape: {current_hidden.shape}")
        
        # Apply final layer norm
        final_hidden = model.apply(
            {'params': params['params']['norm']},
            current_hidden,
            method=lambda module, x: module.norm(x)
        )
        
        # Apply language model head
        final_logits = model.apply(
            {'params': params['params']['lm_head']},
            final_hidden,
            method=lambda module, x: module.lm_head(x)
        )
        
        layer_states['final_logits'] = {
            'hidden_state': np.array(final_logits, dtype=np.float32),
            'shape': final_logits.shape,
            'layer_index': config["num_hidden_layers"] + 1
        }
        
        print(f"JAX final logits shape: {final_logits.shape}")
        
        # Cleanup
        del model, params
        jax.clear_caches()
        gc.collect()
        
        return layer_states, input_ids, None
        
    except Exception as e:
        return None, None, str(e)

def compare_layer_states(pytorch_states, jax_states, tolerance=1e-4):
    """Compare layer states between PyTorch and JAX models"""
    
    print("\n" + "=" * 70)
    print("LAYER-BY-LAYER COMPARISON ANALYSIS")
    print("=" * 70)
    
    comparison_results = []
    
    # Get common layers
    common_layers = set(pytorch_states.keys()) & set(jax_states.keys())
    print(f"Common layers to compare: {len(common_layers)}")
    
    # Sort layers by index for proper order
    sorted_layers = sorted(common_layers, 
                          key=lambda x: pytorch_states[x]['layer_index'])
    
    for layer_name in sorted_layers:
        pt_data = pytorch_states[layer_name]
        jax_data = jax_states[layer_name]
        
        print(f"\n--- Layer: {layer_name} ---")
        print(f"PyTorch shape: {pt_data['shape']}")
        print(f"JAX shape: {jax_data['shape']}")
        
        # Check shapes match
        if pt_data['shape'] != jax_data['shape']:
            print(f"❌ Shape mismatch!")
            comparison_results.append((layer_name, False, "Shape mismatch", 0, 0))
            continue
        
        # Compare hidden states
        pt_hidden = pt_data['hidden_state']
        jax_hidden = jax_data['hidden_state']
        
        abs_diff = np.abs(pt_hidden - jax_hidden)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        
        print(f"Max absolute difference: {max_diff:.2e}")
        print(f"Mean absolute difference: {mean_diff:.2e}")
        print(f"Tolerance threshold: {tolerance:.2e}")
        
        # Check if within tolerance
        within_tolerance = max_diff < tolerance
        
        if within_tolerance:
            print(f"✅ Layer {layer_name} MATCHES")
            status = "MATCH"
        else:
            print(f"❌ Layer {layer_name} DIFFERS")
            status = "DIFFER"
            
            # Show statistics for differing layers
            print(f"  Difference statistics:")
            print(f"    95th percentile: {np.percentile(abs_diff, 95):.2e}")
            print(f"    99th percentile: {np.percentile(abs_diff, 99):.2e}")
            
            # Show worst mismatch location
            worst_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
            print(f"    Worst mismatch at {worst_idx}:")
            print(f"      PyTorch: {pt_hidden[worst_idx]:.6f}")
            print(f"      JAX: {jax_hidden[worst_idx]:.6f}")
        
        comparison_results.append((layer_name, within_tolerance, status, max_diff, mean_diff))
    
    return comparison_results

def analyze_divergence_point(comparison_results):
    """Analyze where divergence first occurs"""
    
    print("\n" + "=" * 70)
    print("DIVERGENCE POINT ANALYSIS")
    print("=" * 70)
    
    first_divergence = None
    divergence_pattern = []
    
    for layer_name, matches, status, max_diff, mean_diff in comparison_results:
        divergence_pattern.append((layer_name, matches, max_diff))
        
        if not matches and first_divergence is None:
            first_divergence = layer_name
    
    print(f"\nDivergence Pattern:")
    for layer_name, matches, max_diff in divergence_pattern:
        status_symbol = "✅" if matches else "❌"
        print(f"  {layer_name:<15} {status_symbol} max_diff: {max_diff:.2e}")
    
    if first_divergence:
        print(f"\n🎯 FIRST DIVERGENCE DETECTED AT: {first_divergence}")
        print(f"All layers before {first_divergence} match within tolerance")
        print(f"Divergence starts at {first_divergence} and may propagate forward")
    else:
        print(f"\n✅ NO DIVERGENCE DETECTED")
        print(f"All layers match within tolerance")
    
    return first_divergence, divergence_pattern

def main():
    """Main test function"""
    print("=" * 70)
    print("PHASE 3.3: LAYER-BY-LAYER INTERMEDIATE ANALYSIS")
    print("=" * 70)
    
    test_input = get_test_input()
    print(f"Test input: '{test_input}'")
    
    # Extract PyTorch layer states
    print(f"\n{'='*50}")
    print("EXTRACTING PYTORCH LAYER STATES")
    print(f"{'='*50}")
    
    pytorch_states, pytorch_tokens, pytorch_error = extract_pytorch_layer_states(test_input)
    if pytorch_error:
        print(f"❌ PyTorch extraction failed: {pytorch_error}")
        return False
    
    print("✅ PyTorch layer states extracted")
    
    # Extract JAX layer states
    print(f"\n{'='*50}")
    print("EXTRACTING JAX LAYER STATES")
    print(f"{'='*50}")
    
    jax_states, jax_tokens, jax_error = extract_jax_layer_states(test_input)
    if jax_error:
        print(f"❌ JAX extraction failed: {jax_error}")
        return False
    
    print("✅ JAX layer states extracted")
    
    # Verify tokens match
    if not np.array_equal(pytorch_tokens, jax_tokens):
        print(f"❌ Token mismatch between models!")
        return False
    
    print(f"✅ Input tokens match: {pytorch_tokens.tolist()}")
    
    # Compare layer states
    comparison_results = compare_layer_states(pytorch_states, jax_states)
    
    # Analyze divergence point
    first_divergence, divergence_pattern = analyze_divergence_point(comparison_results)
    
    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 3.3 FINAL SUMMARY")
    print("=" * 70)
    
    matching_layers = sum(1 for _, matches, _, _, _ in comparison_results if matches)
    total_layers = len(comparison_results)
    
    print(f"Layers analyzed: {total_layers}")
    print(f"Matching layers: {matching_layers}")
    print(f"Differing layers: {total_layers - matching_layers}")
    
    if first_divergence:
        print(f"\n🎯 CRITICAL FINDING:")
        print(f"First divergence occurs at: {first_divergence}")
        print(f"This pinpoints exactly where the issue begins!")
        
        # Provide specific guidance based on where divergence starts
        if first_divergence == "embedding":
            print(f"Issue is in embedding layer - check weight loading!")
        elif first_divergence.startswith("layer_"):
            layer_num = first_divergence.split("_")[1]
            print(f"Issue starts at transformer layer {layer_num}")
            print(f"Focus investigation on attention/MLP in layer {layer_num}")
        elif first_divergence == "final_logits":
            print(f"Issue is in final projection layer")
        
        return False
    else:
        print(f"\n✅ No divergence found - unexpected result!")
        print(f"All layers match but final outputs differ - investigate further")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 