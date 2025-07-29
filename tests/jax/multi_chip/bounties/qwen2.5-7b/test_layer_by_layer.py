#!/usr/bin/env python3
"""
Layer-by-Layer Comparison Test
Focuses on finding the exact layer causing large logits differences.
"""

import os
import json
import numpy as np
import jax
import jax.numpy as jnp
import torch
from qwen_jax_inference import Qwen25ForCausalLM, load_params
from transformers import AutoTokenizer

def test_layer_by_layer_simple():
    """Test layer by layer with minimal memory usage"""
    print("🔍 Layer-by-Layer Simple Comparison Test")
    print("="*60)
    
    # Load config and tokenizer
    config_path = "./weights/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained("./weights")
    
    # Test with simple input
    test_prompt = "Hello"
    inputs = tokenizer(test_prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    
    print(f"Testing with prompt: '{test_prompt}'")
    print(f"Input IDs: {input_ids}")
    
    # Load JAX model and params
    print("\n📥 Loading JAX model...")
    jax_model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
    jax_params = load_params(jax_model, "./weights", jnp.float32)
    
    # JAX forward pass - extract intermediate values
    print("\n🔍 JAX Forward Pass Analysis...")
    
    # 1. Embedding layer
    jax_embeddings = jax_model.apply(
        {'params': jax_params['params']}, 
        jnp.array(input_ids), 
        method=lambda module, ids: module.embed_tokens(ids)
    )
    print(f"JAX Embeddings shape: {jax_embeddings.shape}")
    print(f"JAX Embeddings range: [{float(jnp.min(jax_embeddings)):.6f}, {float(jnp.max(jax_embeddings)):.6f}]")
    print(f"JAX Embeddings mean: {float(jnp.mean(jax_embeddings)):.6f}")
    
    # 2. First layer input norm
    first_layer_norm = jax_model.apply(
        {'params': jax_params['params']}, 
        jax_embeddings,
        method=lambda module, hidden: module.layers[0].input_layernorm(hidden)
    )
    print(f"JAX First Layer Norm shape: {first_layer_norm.shape}")
    print(f"JAX First Layer Norm range: [{float(jnp.min(first_layer_norm)):.6f}, {float(jnp.max(first_layer_norm)):.6f}]")
    print(f"JAX First Layer Norm mean: {float(jnp.mean(first_layer_norm)):.6f}")
    
    # 3. Final layer norm
    # Simulate passing through all layers (we can't do this layer by layer due to memory)
    try:
        # Apply full model to get final hidden states before lm_head
        full_outputs = jax_model.apply(jax_params, input_ids=jnp.array(input_ids))
        if isinstance(full_outputs, dict):
            jax_logits = full_outputs["logits"]
        else:
            jax_logits = full_outputs
        
        print(f"JAX Final Logits shape: {jax_logits.shape}")
        print(f"JAX Final Logits range: [{float(jnp.min(jax_logits)):.6f}, {float(jnp.max(jax_logits)):.6f}]")
        print(f"JAX Final Logits mean: {float(jnp.mean(jax_logits)):.6f}")
        print(f"JAX Final Logits std: {float(jnp.std(jax_logits)):.6f}")
        
        # Check for specific token logits
        last_token_logits = jax_logits[0, -1, :]
        top_5_tokens = jnp.argsort(last_token_logits)[-5:][::-1]
        print("JAX Top 5 tokens and logits:")
        for i, token_id in enumerate(top_5_tokens):
            token_text = tokenizer.decode([int(token_id)])
            logit_val = float(last_token_logits[token_id])
            print(f"  {i+1}. Token {token_id} ('{token_text}'): {logit_val:.6f}")
    
    except Exception as e:
        print(f"❌ JAX full forward pass failed: {e}")
        return False
    
    # Clean up JAX model to free memory
    del jax_model, jax_params
    jax.clear_caches()
    
    print("\n📥 Loading PyTorch model for comparison...")
    try:
        from transformers import AutoModelForCausalLM
        torch_model = AutoModelForCausalLM.from_pretrained(
            "./weights",
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        torch_model.eval()
        
        # PyTorch forward pass
        print("\n🔍 PyTorch Forward Pass Analysis...")
        
        with torch.no_grad():
            input_ids_torch = torch.from_numpy(input_ids).long()
            
            # 1. Embedding layer
            torch_embeddings = torch_model.model.embed_tokens(input_ids_torch)
            torch_emb_np = torch_embeddings.cpu().numpy()
            print(f"PyTorch Embeddings shape: {torch_emb_np.shape}")
            print(f"PyTorch Embeddings range: [{np.min(torch_emb_np):.6f}, {np.max(torch_emb_np):.6f}]")
            print(f"PyTorch Embeddings mean: {np.mean(torch_emb_np):.6f}")
            
            # Compare embeddings
            jax_emb_np = np.array(jax_embeddings)
            emb_diff = np.max(np.abs(jax_emb_np - torch_emb_np))
            print(f"🔍 Embedding Difference: {emb_diff:.2e}")
            
            # 2. First layer input norm
            torch_first_norm = torch_model.model.layers[0].input_layernorm(torch_embeddings)
            torch_norm_np = torch_first_norm.cpu().numpy()
            print(f"PyTorch First Layer Norm shape: {torch_norm_np.shape}")
            print(f"PyTorch First Layer Norm range: [{np.min(torch_norm_np):.6f}, {np.max(torch_norm_np):.6f}]")
            print(f"PyTorch First Layer Norm mean: {np.mean(torch_norm_np):.6f}")
            
            # Compare first layer norm
            first_norm_np = np.array(first_layer_norm)
            norm_diff = np.max(np.abs(first_norm_np - torch_norm_np))
            print(f"🔍 First Layer Norm Difference: {norm_diff:.2e}")
            
            # 3. Full forward pass
            torch_outputs = torch_model(input_ids_torch)
            torch_logits = torch_outputs.logits
            torch_logits_np = torch_logits.cpu().numpy()
            
            print(f"PyTorch Final Logits shape: {torch_logits_np.shape}")
            print(f"PyTorch Final Logits range: [{np.min(torch_logits_np):.6f}, {np.max(torch_logits_np):.6f}]")
            print(f"PyTorch Final Logits mean: {np.mean(torch_logits_np):.6f}")
            print(f"PyTorch Final Logits std: {np.std(torch_logits_np):.6f}")
            
            # Compare final logits
            jax_logits_np = np.array(jax_logits)
            logits_diff = np.max(np.abs(jax_logits_np - torch_logits_np))
            logits_mean_diff = np.mean(np.abs(jax_logits_np - torch_logits_np))
            print(f"🔍 Final Logits Max Difference: {logits_diff:.2e}")
            print(f"🔍 Final Logits Mean Difference: {logits_mean_diff:.2e}")
            
            # Compare top tokens
            torch_last_logits = torch_logits_np[0, -1, :]
            torch_top_5 = np.argsort(torch_last_logits)[-5:][::-1]
            print("PyTorch Top 5 tokens and logits:")
            for i, token_id in enumerate(torch_top_5):
                token_text = tokenizer.decode([int(token_id)])
                logit_val = torch_last_logits[token_id]
                print(f"  {i+1}. Token {token_id} ('{token_text}'): {logit_val:.6f}")
    
    except Exception as e:
        print(f"❌ PyTorch comparison failed: {e}")
        return False
    
    # Final analysis
    print("\n" + "="*60)
    print("🎯 LAYER-BY-LAYER ANALYSIS SUMMARY")
    print("="*60)
    
    issues = []
    if emb_diff > 1e-6:
        issues.append(f"Embedding differences: {emb_diff:.2e}")
    if norm_diff > 1e-6:
        issues.append(f"First layer norm differences: {norm_diff:.2e}")
    if logits_diff > 1e-2:
        issues.append(f"Large final logits differences: {logits_diff:.2e}")
    
    if issues:
        print("❌ Issues Found:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("✅ No major layer-level issues detected")
    
    # Recommendations
    print(f"\n🔧 Analysis:")
    if emb_diff > 1e-6:
        print("   • Embedding layer has differences - check weight loading")
    elif norm_diff > 1e-6:
        print("   • First layer norm has differences - check RMS norm implementation")
    elif logits_diff > norm_diff * 100:
        print("   • Large amplification from norm to final logits - check accumulation through layers")
    else:
        print("   • Differences seem to accumulate gradually through layers")
    
    return logits_diff < 1e-2

if __name__ == "__main__":
    test_layer_by_layer_simple() 