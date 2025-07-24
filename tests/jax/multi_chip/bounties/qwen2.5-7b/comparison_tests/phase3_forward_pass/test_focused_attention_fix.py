#!/usr/bin/env python3
"""
Focused Attention Fix
====================
Quick targeted test to identify and fix the specific attention computation difference.
Based on Phase 3A.1 results: attention output differs by 2.44, needs to be < 1e-6.
"""

import sys
import os
import numpy as np
import json

def test_attention_projection_weights():
    """Test that attention projection weights are loaded correctly after our transpose fix"""
    
    print("=" * 60)
    print("TESTING ATTENTION PROJECTION WEIGHTS")
    print("=" * 60)
    
    # Test PyTorch weights
    try:
        from transformers import AutoModelForCausalLM
        import torch
        
        model_path = "../../weights"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        layer_0_attn = model.model.layers[0].self_attn
        
        pt_q_weight = layer_0_attn.q_proj.weight.detach().float().numpy()
        pt_k_weight = layer_0_attn.k_proj.weight.detach().float().numpy()
        pt_v_weight = layer_0_attn.v_proj.weight.detach().float().numpy()
        pt_o_weight = layer_0_attn.o_proj.weight.detach().float().numpy()
        
        pt_q_bias = layer_0_attn.q_proj.bias.detach().float().numpy()
        pt_k_bias = layer_0_attn.k_proj.bias.detach().float().numpy()
        pt_v_bias = layer_0_attn.v_proj.bias.detach().float().numpy()
        
        print("PyTorch weights:")
        print(f"  Q: {pt_q_weight.shape}, K: {pt_k_weight.shape}, V: {pt_v_weight.shape}, O: {pt_o_weight.shape}")
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"PyTorch loading failed: {e}")
        return False
    
    # Test JAX weights
    try:
        import jax.numpy as jnp
        sys.path.append("../..")
        from qwen_jax_inference import Qwen25ForCausalLM, load_params
        
        with open(f"{model_path}/config.json", 'r') as f:
            config = json.load(f)
        
        model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
        params = load_params(model, model_path, jnp.bfloat16)
        
        layer_0_params = params['params']['layers_0']['self_attn']
        
        jax_q_weight = np.array(layer_0_params['q_proj']['kernel'].astype(jnp.float32))
        jax_k_weight = np.array(layer_0_params['k_proj']['kernel'].astype(jnp.float32))
        jax_v_weight = np.array(layer_0_params['v_proj']['kernel'].astype(jnp.float32))
        jax_o_weight = np.array(layer_0_params['o_proj']['kernel'].astype(jnp.float32))
        
        jax_q_bias = np.array(layer_0_params['q_proj']['bias'].astype(jnp.float32))
        jax_k_bias = np.array(layer_0_params['k_proj']['bias'].astype(jnp.float32))
        jax_v_bias = np.array(layer_0_params['v_proj']['bias'].astype(jnp.float32))
        
        print("JAX weights:")
        print(f"  Q: {jax_q_weight.shape}, K: {jax_k_weight.shape}, V: {jax_v_weight.shape}, O: {jax_o_weight.shape}")
        
        del model, params
        
    except Exception as e:
        print(f"JAX loading failed: {e}")
        return False
    
    # Compare weights
    print("\nWeight comparison:")
    
    # ALL projection weights should be transposed for JAX Dense compatibility
    q_diff = np.abs(pt_q_weight.T - jax_q_weight).max()
    print(f"Q weight diff (PyTorch.T vs JAX): {q_diff:.8e}")
    
    # K projection (JAX should be transpose of PyTorch)
    k_diff = np.abs(pt_k_weight.T - jax_k_weight).max()
    print(f"K weight diff (PyTorch.T vs JAX): {k_diff:.8e}")
    
    # V projection (JAX should be transpose of PyTorch)
    v_diff = np.abs(pt_v_weight.T - jax_v_weight).max()
    print(f"V weight diff (PyTorch.T vs JAX): {v_diff:.8e}")
    
    # O projection (JAX should be transpose of PyTorch)
    o_diff = np.abs(pt_o_weight.T - jax_o_weight).max()
    print(f"O weight diff (PyTorch.T vs JAX): {o_diff:.8e}")
    
    # Biases (should be identical)
    q_bias_diff = np.abs(pt_q_bias - jax_q_bias).max()
    k_bias_diff = np.abs(pt_k_bias - jax_k_bias).max()
    v_bias_diff = np.abs(pt_v_bias - jax_v_bias).max()
    print(f"Q bias diff: {q_bias_diff:.8e}")
    print(f"K bias diff: {k_bias_diff:.8e}")
    print(f"V bias diff: {v_bias_diff:.8e}")
    
    # Check if all weights match correctly
    weights_ok = (
        q_diff < 1e-6 and
        k_diff < 1e-6 and
        v_diff < 1e-6 and
        o_diff < 1e-6 and
        q_bias_diff < 1e-6 and
        k_bias_diff < 1e-6 and
        v_bias_diff < 1e-6
    )
    
    if weights_ok:
        print("✅ All attention weights match correctly!")
        return True
    else:
        print("❌ Some attention weights don't match")
        return False

def test_simple_projection_computation():
    """Test Q/K/V projection computation with a simple input"""
    
    print("\n" + "=" * 60)
    print("TESTING SIMPLE PROJECTION COMPUTATION")
    print("=" * 60)
    
    # Create a simple test input
    test_input = np.random.randn(1, 1, 3584).astype(np.float32)
    print(f"Test input shape: {test_input.shape}")
    
    # Load PyTorch model and compute projections
    try:
        from transformers import AutoModelForCausalLM
        import torch
        
        model_path = "../../weights"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        layer_0_attn = model.model.layers[0].self_attn
        
        # Convert input to PyTorch tensor
        pt_input = torch.from_numpy(test_input).to(torch.bfloat16)
        
        with torch.no_grad():
            pt_q = layer_0_attn.q_proj(pt_input).float().numpy()
            pt_k = layer_0_attn.k_proj(pt_input).float().numpy()
            pt_v = layer_0_attn.v_proj(pt_input).float().numpy()
        
        print(f"PyTorch projections: Q{pt_q.shape}, K{pt_k.shape}, V{pt_v.shape}")
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"PyTorch computation failed: {e}")
        return False
    
    # Load JAX model and compute projections
    try:
        import jax.numpy as jnp
        sys.path.append("../..")
        from qwen_jax_inference import Qwen25ForCausalLM, load_params
        
        with open(f"{model_path}/config.json", 'r') as f:
            config = json.load(f)
        
        model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
        params = load_params(model, model_path, jnp.bfloat16)
        
        layer_0_params = params['params']['layers_0']['self_attn']
        
        # Manual JAX computation
        jax_input = jnp.array(test_input).astype(jnp.bfloat16)
        
        jax_q = jnp.dot(jax_input, layer_0_params['q_proj']['kernel']) + layer_0_params['q_proj']['bias']
        jax_k = jnp.dot(jax_input, layer_0_params['k_proj']['kernel']) + layer_0_params['k_proj']['bias']
        jax_v = jnp.dot(jax_input, layer_0_params['v_proj']['kernel']) + layer_0_params['v_proj']['bias']
        
        jax_q = np.array(jax_q.astype(jnp.float32))
        jax_k = np.array(jax_k.astype(jnp.float32))
        jax_v = np.array(jax_v.astype(jnp.float32))
        
        print(f"JAX projections: Q{jax_q.shape}, K{jax_k.shape}, V{jax_v.shape}")
        
        del model, params
        
    except Exception as e:
        print(f"JAX computation failed: {e}")
        return False
    
    # Compare results
    print("\nProjection computation comparison:")
    q_diff = np.abs(pt_q - jax_q).max()
    k_diff = np.abs(pt_k - jax_k).max()
    v_diff = np.abs(pt_v - jax_v).max()
    
    print(f"Q projection diff: {q_diff:.8e}")
    print(f"K projection diff: {k_diff:.8e}")
    print(f"V projection diff: {v_diff:.8e}")
    
    if q_diff < 1e-6 and k_diff < 1e-6 and v_diff < 1e-6:
        print("✅ All projections match perfectly!")
        return True
    else:
        print("❌ Projection computations differ")
        print(f"Sample Q diff: {(pt_q - jax_q).flat[:5]}")
        print(f"Sample K diff: {(pt_k - jax_k).flat[:5]}")
        print(f"Sample V diff: {(pt_v - jax_v).flat[:5]}")
        return False

def main():
    """Run focused attention tests"""
    print("FOCUSED ATTENTION MECHANISM TESTING")
    print("Goal: Identify specific cause of 2.44 attention output difference")
    
    # Test 1: Verify weights are loaded correctly
    weights_ok = test_attention_projection_weights()
    
    if not weights_ok:
        print("\n❌ ISSUE: Weights not loaded correctly")
        return False
    
    # Test 2: Test projection computation
    projections_ok = test_simple_projection_computation()
    
    if not projections_ok:
        print("\n❌ ISSUE: Projection computation differs")
        return False
    
    print("\n🎉 SUCCESS: Weights and projections work correctly!")
    print("📋 Next: Need to investigate attention weights computation or GQA")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 