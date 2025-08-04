#!/usr/bin/env python3
"""
Side-by-side comparison of original vs fully parallel model
"""
import os
import jax
import jax.numpy as jnp
import json

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

# Import both implementations
import q25j7_tensor_parallel_fixed
from q25j7_tensor_parallel_fixed import Qwen25ForCausalLM as ParallelModel, load_params, setup_device_mesh

from q25j7_tensor_parallel import Qwen25ForCausalLM as OriginalModel, load_params as original_load_params
from transformers import AutoTokenizer

def compare_models():
    """Compare original vs fully parallel model side-by-side"""
    
    # Setup mesh for parallel model
    mesh = setup_device_mesh()
    q25j7_tensor_parallel_fixed.mesh = mesh
    
    with open("weights/config.json") as f:
        config = json.load(f)
    
    print("=== SIDE-BY-SIDE MODEL COMPARISON ===\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("weights")
    
    # Test input
    test_input = "What is 2 + 2?"
    input_ids = tokenizer.encode(test_input, return_tensors="jax")
    
    print(f"Input: '{test_input}'")
    print(f"Input IDs: {input_ids}")
    
    # Test 1: Original Model
    print("\n--- ORIGINAL MODEL (working) ---")
    try:
        original_model = OriginalModel(config=config, dtype=jnp.bfloat16)
        original_params = original_load_params(original_model, "weights", jnp.bfloat16)
        
        with mesh:  # Original model also uses the mesh for MLP layers
            original_outputs = original_model.apply(original_params, input_ids, return_dict=True)
            original_logits = original_outputs['logits'][0, -1, :]
        
        print(f"Original logits min/max: {float(jnp.min(original_logits)):.4f}, {float(jnp.max(original_logits)):.4f}")
        print(f"Original logits mean/std: {float(jnp.mean(original_logits)):.4f}, {float(jnp.std(original_logits)):.4f}")
        
        original_top = jnp.argsort(original_logits)[-5:][::-1]
        print("Original top 5 tokens:")
        for i, token_id in enumerate(original_top):
            token_text = tokenizer.decode(int(token_id))
            prob = float(jax.nn.softmax(original_logits)[token_id])
            print(f"  {i+1}. Token {token_id}: '{token_text}' (prob: {prob:.4f})")
        
    except Exception as e:
        print(f"❌ Original model failed: {e}")
        original_logits = None
    
    # Test 2: Fully Parallel Model
    print("\n--- FULLY PARALLEL MODEL (testing) ---")
    try:
        parallel_model = ParallelModel(config=config, dtype=jnp.bfloat16)
        parallel_params = load_params(parallel_model, "weights", jnp.bfloat16)
        
        with mesh:
            parallel_outputs = parallel_model.apply(parallel_params, input_ids, return_dict=True)
            parallel_logits = parallel_outputs['logits'][0, -1, :]
        
        print(f"Parallel logits min/max: {float(jnp.min(parallel_logits)):.4f}, {float(jnp.max(parallel_logits)):.4f}")
        print(f"Parallel logits mean/std: {float(jnp.mean(parallel_logits)):.4f}, {float(jnp.std(parallel_logits)):.4f}")
        
        parallel_top = jnp.argsort(parallel_logits)[-5:][::-1]
        print("Parallel top 5 tokens:")
        for i, token_id in enumerate(parallel_top):
            token_text = tokenizer.decode(int(token_id))
            prob = float(jax.nn.softmax(parallel_logits)[token_id])
            print(f"  {i+1}. Token {token_id}: '{token_text}' (prob: {prob:.4f})")
        
    except Exception as e:
        print(f"❌ Parallel model failed: {e}")
        parallel_logits = None
    
    # Compare if both worked
    if original_logits is not None and parallel_logits is not None:
        print("\n--- DETAILED COMPARISON ---")
        
        # Check if logits are close
        max_diff = float(jnp.max(jnp.abs(original_logits - parallel_logits)))
        mean_diff = float(jnp.mean(jnp.abs(original_logits - parallel_logits)))
        
        print(f"Max logit difference: {max_diff:.6f}")
        print(f"Mean logit difference: {mean_diff:.6f}")
        
        if max_diff < 0.1:
            print("✅ Logits are very close!")
        elif max_diff < 1.0:
            print("⚠️ Logits are somewhat close")
        else:
            print("❌ Logits differ significantly")
        
        # Check specific important tokens
        important_tokens = [16, 17, 220]  # '1', '2', ' '
        print("\nComparison of important tokens:")
        for token_id in important_tokens:
            token_text = tokenizer.decode(token_id)
            orig_prob = float(jax.nn.softmax(original_logits)[token_id])
            par_prob = float(jax.nn.softmax(parallel_logits)[token_id])
            diff = abs(orig_prob - par_prob)
            print(f"  Token {token_id} ('{token_text}'): Original {orig_prob:.4f}, Parallel {par_prob:.4f}, Diff {diff:.4f}")
        
        # Check if top tokens are the same
        orig_top1 = int(jnp.argmax(original_logits))
        par_top1 = int(jnp.argmax(parallel_logits))
        
        if orig_top1 == par_top1:
            print(f"✅ Both models predict same top token: {orig_top1}")
        else:
            print(f"❌ Different top tokens: Original {orig_top1}, Parallel {par_top1}")

if __name__ == "__main__":
    compare_models()