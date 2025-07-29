#!/usr/bin/env python3
"""
Debug Logits Differences - Diagnostic Tool
Identifies the root cause of logits differences between JAX and PyTorch implementations.
"""

import os
import sys
import json
import numpy as np
import torch
import jax
import jax.numpy as jnp
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import our JAX implementation
from qwen_jax_inference import Qwen25ForCausalLM, load_params

class LogitsDifferenceDebugger:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.jax_model = None
        self.jax_params = None
        self.torch_model = None
        self.tokenizer = None
        
    def load_models(self):
        """Load both models for comparison"""
        print("🔍 Loading models for debugging...")
        
        # Load config
        config_path = os.path.join(self.model_path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load JAX model
        print("Loading JAX model...")
        self.jax_model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)  # Use float32 for precision
        self.jax_params = load_params(self.jax_model, self.model_path, jnp.float32)
        
        # Load PyTorch model
        print("Loading PyTorch model...")
        self.torch_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,  # Use float32 for precision
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.torch_model.eval()
        
        print("✅ Models loaded")
    
    def compare_embeddings(self, input_ids):
        """Compare embedding outputs"""
        print("\n🔍 Debugging embeddings...")
        
        # JAX embeddings
        jax_embeddings = self.jax_model.apply(
            {'params': self.jax_params['params']}, 
            input_ids, 
            method=lambda module, ids: module.embed_tokens(ids)
        )
        
        # PyTorch embeddings
        torch_embeddings = self.torch_model.model.embed_tokens(torch.from_numpy(np.array(input_ids)).long())
        
        # Compare
        jax_emb_np = np.array(jax_embeddings)
        torch_emb_np = torch_embeddings.detach().cpu().numpy()
        
        emb_diff = np.max(np.abs(jax_emb_np - torch_emb_np))
        emb_mean_diff = np.mean(np.abs(jax_emb_np - torch_emb_np))
        
        print(f"   Embedding max diff: {emb_diff:.2e}")
        print(f"   Embedding mean diff: {emb_mean_diff:.2e}")
        print(f"   JAX embedding shape: {jax_emb_np.shape}")
        print(f"   PyTorch embedding shape: {torch_emb_np.shape}")
        
        return {
            'max_diff': emb_diff,
            'mean_diff': emb_mean_diff,
            'jax_embeddings': jax_embeddings,
            'torch_embeddings': torch_embeddings
        }
    
    def compare_first_layer_components(self, hidden_states_jax, hidden_states_torch):
        """Compare first layer components step by step"""
        print("\n🔍 Debugging first layer components...")
        
        # Get first layer parameters
        layer_0_jax = self.jax_params['params']['layers_0']
        layer_0_torch = self.torch_model.model.layers[0]
        
        batch, seq, hidden_size = hidden_states_jax.shape
        
        # 1. Input LayerNorm
        print("   Testing input layernorm...")
        
        # JAX RMS Norm
        eps = 1e-6
        jax_hidden = hidden_states_jax
        variance = jnp.mean(jax_hidden**2, axis=-1, keepdims=True)
        jax_normed = jax_hidden * jnp.power(variance + eps, -0.5)
        jax_normed = layer_0_jax['input_layernorm']['scale'] * jax_normed
        
        # PyTorch RMS Norm
        torch_hidden = hidden_states_torch
        torch_normed = layer_0_torch.input_layernorm(torch_hidden)
        
        norm_diff = np.max(np.abs(np.array(jax_normed) - torch_normed.detach().cpu().numpy()))
        print(f"   Input norm max diff: {norm_diff:.2e}")
        
        # 2. Attention projections
        print("   Testing attention projections...")
        
        # JAX projections
        jax_q = jnp.dot(jax_normed, layer_0_jax['self_attn']['q_proj']['kernel']) + layer_0_jax['self_attn']['q_proj']['bias']
        jax_k = jnp.dot(jax_normed, layer_0_jax['self_attn']['k_proj']['kernel']) + layer_0_jax['self_attn']['k_proj']['bias']
        jax_v = jnp.dot(jax_normed, layer_0_jax['self_attn']['v_proj']['kernel']) + layer_0_jax['self_attn']['v_proj']['bias']
        
        # PyTorch projections
        torch_q = layer_0_torch.self_attn.q_proj(torch_normed)
        torch_k = layer_0_torch.self_attn.k_proj(torch_normed)
        torch_v = layer_0_torch.self_attn.v_proj(torch_normed)
        
        q_diff = np.max(np.abs(np.array(jax_q) - torch_q.detach().cpu().numpy()))
        k_diff = np.max(np.abs(np.array(jax_k) - torch_k.detach().cpu().numpy()))
        v_diff = np.max(np.abs(np.array(jax_v) - torch_v.detach().cpu().numpy()))
        
        print(f"   Q projection max diff: {q_diff:.2e}")
        print(f"   K projection max diff: {k_diff:.2e}")
        print(f"   V projection max diff: {v_diff:.2e}")
        
        # 3. Check weight shapes and values
        print("   Checking weight alignment...")
        
        jax_q_weight = layer_0_jax['self_attn']['q_proj']['kernel']
        torch_q_weight = layer_0_torch.self_attn.q_proj.weight
        
        print(f"   JAX Q weight shape: {jax_q_weight.shape}")
        print(f"   PyTorch Q weight shape: {torch_q_weight.shape}")
        
        # Compare weights (need to transpose PyTorch weights)
        torch_q_weight_transposed = torch_q_weight.T.detach().cpu().numpy()
        weight_diff = np.max(np.abs(np.array(jax_q_weight) - torch_q_weight_transposed))
        print(f"   Q weight alignment diff: {weight_diff:.2e}")
        
        return {
            'norm_diff': norm_diff,
            'q_diff': q_diff,
            'k_diff': k_diff,
            'v_diff': v_diff,
            'weight_diff': weight_diff
        }
    
    def debug_prompt(self, prompt: str):
        """Debug a specific prompt end-to-end"""
        print(f"\n🔍 Debugging prompt: '{prompt}'")
        print("="*60)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"]
        
        print(f"Input IDs: {input_ids}")
        print(f"Tokens: {[self.tokenizer.decode([id]) for id in input_ids[0]]}")
        
        # 1. Compare embeddings
        emb_results = self.compare_embeddings(input_ids)
        
        if emb_results['max_diff'] > 1e-6:
            print("❌ ISSUE: Large embedding differences detected!")
            return
        
        # 2. Compare first layer
        first_layer_results = self.compare_first_layer_components(
            emb_results['jax_embeddings'], 
            emb_results['torch_embeddings']
        )
        
        # 3. Full forward pass comparison
        print("\n🔍 Full forward pass comparison...")
        
        # JAX forward pass
        jax_outputs = self.jax_model.apply(self.jax_params, input_ids=jnp.array(input_ids))
        jax_logits = jax_outputs if not isinstance(jax_outputs, dict) else jax_outputs["logits"]
        
        # PyTorch forward pass
        with torch.no_grad():
            torch_outputs = self.torch_model(torch.from_numpy(input_ids).long())
            torch_logits = torch_outputs.logits
        
        # Compare final logits
        jax_logits_np = np.array(jax_logits)
        torch_logits_np = torch_logits.detach().cpu().numpy()
        
        final_diff = np.max(np.abs(jax_logits_np - torch_logits_np))
        final_mean_diff = np.mean(np.abs(jax_logits_np - torch_logits_np))
        
        print(f"   Final logits max diff: {final_diff:.2e}")
        print(f"   Final logits mean diff: {final_mean_diff:.2e}")
        
        # Find worst differing tokens
        diff_matrix = np.abs(jax_logits_np - torch_logits_np)
        worst_indices = np.unravel_index(np.argmax(diff_matrix), diff_matrix.shape)
        
        print(f"   Worst difference at position {worst_indices}: {diff_matrix[worst_indices]:.2e}")
        
        # Sample a few logits values for comparison
        last_token_logits_jax = jax_logits_np[0, -1, :]
        last_token_logits_torch = torch_logits_np[0, -1, :]
        
        top_5_jax = np.argsort(last_token_logits_jax)[-5:][::-1]
        top_5_torch = np.argsort(last_token_logits_torch)[-5:][::-1]
        
        print("   Top 5 tokens (JAX):")
        for i, token_id in enumerate(top_5_jax):
            token_text = self.tokenizer.decode([token_id])
            print(f"     {i+1}. Token {token_id} ('{token_text}'): {last_token_logits_jax[token_id]:.3f}")
        
        print("   Top 5 tokens (PyTorch):")
        for i, token_id in enumerate(top_5_torch):
            token_text = self.tokenizer.decode([token_id])
            print(f"     {i+1}. Token {token_id} ('{token_text}'): {last_token_logits_torch[token_id]:.3f}")
        
        return {
            'embedding_results': emb_results,
            'first_layer_results': first_layer_results,
            'final_logits_diff': final_diff,
            'final_logits_mean_diff': final_mean_diff
        }

def main():
    """Main debugging function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug Logits Differences")
    parser.add_argument("--model_path", type=str, default="./weights", help="Path to model weights")
    parser.add_argument("--prompt", type=str, default="Hello", help="Prompt to debug")
    args = parser.parse_args()
    
    print("🚀 Starting Logits Difference Debugging")
    
    debugger = LogitsDifferenceDebugger(args.model_path)
    debugger.load_models()
    
    # Debug the specified prompt
    results = debugger.debug_prompt(args.prompt)
    
    print("\n" + "="*60)
    print("🎯 DEBUGGING SUMMARY")
    print("="*60)
    
    if results:
        emb_diff = results['embedding_results']['max_diff']
        layer_diff = results['first_layer_results']['weight_diff']
        final_diff = results['final_logits_diff']
        
        print(f"📊 Key Metrics:")
        print(f"   • Embedding difference: {emb_diff:.2e}")
        print(f"   • First layer weight alignment: {layer_diff:.2e}")
        print(f"   • Final logits difference: {final_diff:.2e}")
        
        # Identify likely issues
        issues = []
        if emb_diff > 1e-6:
            issues.append("Embedding layer misalignment")
        if layer_diff > 1e-6:
            issues.append("Weight loading/transpose issues")
        if final_diff > layer_diff * 10:
            issues.append("Accumulating errors through layers")
        
        if issues:
            print(f"\n⚠️ Likely Issues:")
            for issue in issues:
                print(f"   • {issue}")
        else:
            print(f"\n✅ No obvious issues identified in component analysis")
    
    print("\n🔧 Next Steps:")
    print("   1. Fix any weight loading/transpose issues")
    print("   2. Verify numerical precision settings")
    print("   3. Check for implementation differences in layers")
    print("   4. Run layer-by-layer comparison")

if __name__ == "__main__":
    main() 