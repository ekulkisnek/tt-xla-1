#!/usr/bin/env python3
"""
Debug Weight Loading - Lightweight Diagnostic
Focus on weight loading, transpose, and mapping issues without loading full models.
"""

import os
import json
import numpy as np
import torch
import jax.numpy as jnp
from safetensors import safe_open
from transformers import AutoTokenizer

# Import weight loading functions
from qwen_jax_inference import get_param_path, transpose_if_needed, process_safetensors_file

class WeightLoadingDebugger:
    def __init__(self, model_path: str):
        self.model_path = model_path
        
    def compare_weight_loading(self):
        """Compare how weights are loaded between our JAX and PyTorch approach"""
        print("🔍 Analyzing weight loading differences...")
        
        # Load a single safetensors file for analysis
        safetensors_files = [f for f in os.listdir(self.model_path) if f.endswith('.safetensors')]
        test_file = os.path.join(self.model_path, safetensors_files[0])
        
        print(f"Analyzing file: {test_file}")
        
        # Analyze weight mapping and transpose operations
        weight_analysis = {}
        
        with safe_open(test_file, framework="pt") as f:
            for key in list(f.keys())[:10]:  # Analyze first 10 weights
                print(f"\n🔍 Analyzing weight: {key}")
                
                # Get param path for our JAX implementation
                param_path = get_param_path(key)
                
                if param_path is None:
                    print(f"   ❌ No mapping found for {key}")
                    continue
                
                # Load original PyTorch weight
                original_weight = f.get_tensor(key)
                if hasattr(original_weight, 'detach'):
                    original_weight = original_weight.detach().cpu().numpy()
                
                # Apply our transpose logic
                transposed_weight = transpose_if_needed(key, jnp.array(original_weight))
                transposed_weight_np = np.array(transposed_weight)
                
                # Check if transpose actually happened
                transpose_occurred = not np.array_equal(original_weight, transposed_weight_np)
                
                weight_analysis[key] = {
                    'param_path': param_path,
                    'original_shape': original_weight.shape,
                    'transposed_shape': transposed_weight_np.shape,
                    'transpose_occurred': transpose_occurred,
                    'is_projection': 'proj' in key or 'lm_head' in key,
                    'should_transpose': 'weight' in key and ('proj' in key or 'lm_head' in key)
                }
                
                print(f"   Original shape: {original_weight.shape}")
                print(f"   Transposed shape: {transposed_weight_np.shape}")
                print(f"   Transpose occurred: {transpose_occurred}")
                print(f"   Should transpose: {weight_analysis[key]['should_transpose']}")
                
                # Check for potential issues
                if weight_analysis[key]['should_transpose'] and not transpose_occurred:
                    print(f"   ⚠️ WARNING: Expected transpose but didn't occur!")
                elif not weight_analysis[key]['should_transpose'] and transpose_occurred:
                    print(f"   ⚠️ WARNING: Unexpected transpose occurred!")
                else:
                    print(f"   ✅ Transpose behavior correct")
        
        return weight_analysis
    
    def check_embedding_weight_tying(self):
        """Check if embedding and lm_head weights are properly tied"""
        print("\n🔍 Checking embedding weight tying...")
        
        # Load the JAX processed weights
        jax_params = process_safetensors_file(
            os.path.join(self.model_path, "model-00001-of-00004.safetensors"), 
            dtype=jnp.float32
        )
        
        embed_weight = jax_params['params']['embed_tokens']['embedding']
        lm_head_weight = jax_params['params']['lm_head']['kernel']
        
        print(f"Embedding weight shape: {embed_weight.shape}")
        print(f"LM head weight shape: {lm_head_weight.shape}")
        
        # Check if they should be tied (same shape)
        if embed_weight.shape == lm_head_weight.shape:
            max_diff = jnp.max(jnp.abs(embed_weight - lm_head_weight))
            print(f"Max difference between embed and lm_head: {float(max_diff):.2e}")
            
            if float(max_diff) < 1e-6:
                print("✅ Weights are properly tied")
                return True
            else:
                print("❌ Weights are NOT tied (this may be the issue!)")
                return False
        else:
            print("❌ Embedding and LM head have different shapes!")
            return False
    
    def analyze_attention_weight_shapes(self):
        """Analyze attention projection weight shapes"""
        print("\n🔍 Analyzing attention weight shapes...")
        
        # Load config to understand expected shapes
        config_path = os.path.join(self.model_path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        hidden_size = config['hidden_size']
        num_attention_heads = config['num_attention_heads']
        num_key_value_heads = config.get('num_key_value_heads', num_attention_heads)
        head_dim = hidden_size // num_attention_heads
        
        print(f"Config: hidden_size={hidden_size}, num_heads={num_attention_heads}, num_kv_heads={num_key_value_heads}")
        
        # Expected shapes after our transpose
        expected_shapes = {
            'q_proj': (hidden_size, hidden_size),
            'k_proj': (hidden_size, num_key_value_heads * head_dim),
            'v_proj': (hidden_size, num_key_value_heads * head_dim),
            'o_proj': (hidden_size, hidden_size)
        }
        
        # Load first layer weights
        jax_params = process_safetensors_file(
            os.path.join(self.model_path, "model-00001-of-00004.safetensors"), 
            dtype=jnp.float32
        )
        
        # Check if we have the first layer
        if 'layers_0' in jax_params['params']:
            layer_0 = jax_params['params']['layers_0']['self_attn']
            
            for proj_name, expected_shape in expected_shapes.items():
                if proj_name in layer_0:
                    actual_shape = layer_0[proj_name]['kernel'].shape
                    matches = actual_shape == expected_shape
                    
                    print(f"   {proj_name}: expected {expected_shape}, actual {actual_shape} {'✅' if matches else '❌'}")
                    
                    if not matches:
                        print(f"      ⚠️ Shape mismatch detected!")
                        return False
                else:
                    print(f"   {proj_name}: NOT FOUND")
                    return False
            
            print("✅ All attention projection shapes correct")
            return True
        else:
            print("❌ Could not find layers_0 in parameters")
            return False
    
    def compare_with_pytorch_reference(self):
        """Load a small PyTorch reference for comparison"""
        print("\n🔍 Loading PyTorch reference for comparison...")
        
        # Load tokenizer and a single input
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        test_input = "Hello"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        # Just load the PyTorch embedding layer for comparison
        try:
            from transformers import AutoModelForCausalLM
            
            # Load only embeddings for memory efficiency
            torch_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Get embeddings
            torch_embeddings = torch_model.model.embed_tokens(inputs['input_ids'])
            torch_emb_values = torch_embeddings[0, 0, :5].detach().cpu().numpy()  # First 5 values
            
            print(f"PyTorch embedding sample: {torch_emb_values}")
            
            # Compare with our JAX embeddings
            jax_params = process_safetensors_file(
                os.path.join(self.model_path, "model-00001-of-00004.safetensors"), 
                dtype=jnp.float32
            )
            
            token_id = inputs['input_ids'][0, 0].item()
            jax_emb_values = np.array(jax_params['params']['embed_tokens']['embedding'][token_id, :5])
            
            print(f"JAX embedding sample: {jax_emb_values}")
            
            emb_diff = np.max(np.abs(torch_emb_values - jax_emb_values))
            print(f"Embedding difference: {emb_diff:.2e}")
            
            del torch_model  # Clean up
            
            return emb_diff < 1e-6
            
        except Exception as e:
            print(f"Could not load PyTorch model: {e}")
            return None

def main():
    """Main debugging function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug Weight Loading")
    parser.add_argument("--model_path", type=str, default="./weights", help="Path to model weights")
    args = parser.parse_args()
    
    print("🚀 Starting Weight Loading Analysis")
    
    debugger = WeightLoadingDebugger(args.model_path)
    
    # Run analyses
    print("\n" + "="*60)
    print("WEIGHT LOADING ANALYSIS")
    print("="*60)
    
    # 1. Weight mapping and transpose analysis
    weight_analysis = debugger.compare_weight_loading()
    
    # 2. Check embedding weight tying
    tying_correct = debugger.check_embedding_weight_tying()
    
    # 3. Check attention shapes
    shapes_correct = debugger.analyze_attention_weight_shapes()
    
    # 4. PyTorch comparison (if memory allows)
    embedding_correct = debugger.compare_with_pytorch_reference()
    
    # Summary
    print("\n" + "="*60)
    print("🎯 WEIGHT LOADING SUMMARY")
    print("="*60)
    
    issues_found = []
    
    if not tying_correct:
        issues_found.append("Embedding weight tying issue")
    
    if not shapes_correct:
        issues_found.append("Attention projection shape mismatch")
    
    if embedding_correct is False:
        issues_found.append("Embedding value mismatch with PyTorch")
    
    # Check transpose issues from weight analysis
    transpose_issues = 0
    for key, analysis in weight_analysis.items():
        if analysis['should_transpose'] != analysis['transpose_occurred']:
            transpose_issues += 1
    
    if transpose_issues > 0:
        issues_found.append(f"Transpose issues in {transpose_issues} weights")
    
    if issues_found:
        print("❌ Issues Found:")
        for issue in issues_found:
            print(f"   • {issue}")
    else:
        print("✅ No obvious weight loading issues detected")
    
    print(f"\n🔧 Recommendations:")
    if not tying_correct:
        print("   • Fix embedding weight tying in model definition")
    if not shapes_correct:
        print("   • Check attention projection weight loading and transpose logic")
    if embedding_correct is False:
        print("   • Verify embedding layer weight mapping")
    if transpose_issues > 0:
        print("   • Review transpose_if_needed function logic")
    
    if not issues_found:
        print("   • Weight loading appears correct - check model architecture implementation")

if __name__ == "__main__":
    main() 