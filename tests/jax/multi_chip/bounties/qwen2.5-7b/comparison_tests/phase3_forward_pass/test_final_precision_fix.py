#!/usr/bin/env python3
"""
Final Precision Fix
==================
Target the remaining 0.1-3 logits differences to achieve 1e-6 tolerance.
We've already achieved 90-95% improvement, now targeting final precision issues.
"""

import sys
import os
import numpy as np
import json

def check_rmsnorm_implementation():
    """Check if RMS norm implementation matches exactly between PyTorch and JAX"""
    
    print("=" * 60)
    print("CHECKING RMS NORM IMPLEMENTATION")
    print("=" * 60)
    
    # Test data
    test_input = np.random.randn(1, 1, 3584).astype(np.float32)
    test_weight = np.random.randn(3584).astype(np.float32)
    eps = 1e-6
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Test weight shape: {test_weight.shape}")
    print(f"Epsilon: {eps}")
    
    # PyTorch RMS norm
    try:
        import torch
        
        def pytorch_rmsnorm(x, weight, eps=1e-6):
            input_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + eps)
            return weight * x.to(input_dtype)
        
        pt_input = torch.from_numpy(test_input)
        pt_weight = torch.from_numpy(test_weight)
        
        pt_result = pytorch_rmsnorm(pt_input, pt_weight, eps)
        pt_result_np = pt_result.numpy()
        
        print(f"PyTorch RMS norm result shape: {pt_result_np.shape}")
        
    except Exception as e:
        print(f"PyTorch RMS norm failed: {e}")
        return False
    
    # JAX RMS norm
    try:
        import jax.numpy as jnp
        
        def jax_rmsnorm(x, weight, eps=1e-6):
            input_dtype = x.dtype
            x = x.astype(jnp.float32)
            variance = jnp.mean(x**2, axis=-1, keepdims=True)
            x = x * jnp.power(variance + eps, -0.5)  # rsqrt equivalent
            return weight * x.astype(input_dtype)
        
        jax_input = jnp.array(test_input)
        jax_weight = jnp.array(test_weight)
        
        jax_result = jax_rmsnorm(jax_input, jax_weight, eps)
        jax_result_np = np.array(jax_result)
        
        print(f"JAX RMS norm result shape: {jax_result_np.shape}")
        
    except Exception as e:
        print(f"JAX RMS norm failed: {e}")
        return False
    
    # Compare
    diff = np.abs(pt_result_np - jax_result_np).max()
    print(f"RMS norm max difference: {diff:.8e}")
    
    if diff < 1e-6:
        print("✅ RMS norm implementations match perfectly")
        return True
    else:
        print("❌ RMS norm implementations differ")
        print(f"Sample PyTorch: {pt_result_np.flat[:5]}")
        print(f"Sample JAX: {jax_result_np.flat[:5]}")
        return False

def check_dtype_precision():
    """Check dtype handling and precision in the models"""
    
    print("\n" + "=" * 60)
    print("CHECKING DTYPE PRECISION")
    print("=" * 60)
    
    # Test PyTorch model dtype
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
        
        layer_0 = model.model.layers[0]
        
        print("PyTorch model dtypes:")
        print(f"  Q proj weight: {layer_0.self_attn.q_proj.weight.dtype}")
        print(f"  Input layernorm weight: {layer_0.input_layernorm.weight.dtype}")
        
        # Check RMS norm epsilon
        if hasattr(layer_0.input_layernorm, 'eps'):
            print(f"  RMS norm eps: {layer_0.input_layernorm.eps}")
        elif hasattr(layer_0.input_layernorm, 'variance_epsilon'):
            print(f"  RMS norm eps: {layer_0.input_layernorm.variance_epsilon}")
        else:
            print("  RMS norm eps: not found in attributes")
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"PyTorch dtype check failed: {e}")
        return False
    
    # Test JAX model dtype
    try:
        import jax.numpy as jnp
        sys.path.append("../..")
        from qwen_jax_inference import Qwen25ForCausalLM, load_params
        
        with open(f"{model_path}/config.json", 'r') as f:
            config = json.load(f)
        
        model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
        params = load_params(model, model_path, jnp.bfloat16)
        
        layer_0_params = params['params']['layers_0']
        
        print("JAX model dtypes:")
        print(f"  Q proj weight: {layer_0_params['self_attn']['q_proj']['kernel'].dtype}")
        print(f"  Input layernorm weight: {layer_0_params['input_layernorm']['scale'].dtype}")
        print(f"  RMS norm eps from config: {config.get('rms_norm_eps', 'not found')}")
        
        del model, params
        
    except Exception as e:
        print(f"JAX dtype check failed: {e}")
        return False
    
    print("✅ Dtype precision check completed")
    return True

def test_small_attention_computation():
    """Test attention computation with a tiny example to isolate precision issues"""
    
    print("\n" + "=" * 60)
    print("TESTING SMALL ATTENTION COMPUTATION")
    print("=" * 60)
    
    # Create minimal test case
    batch_size, seq_len, hidden_size = 1, 1, 32
    num_heads, num_kv_heads = 4, 2
    head_dim = hidden_size // num_heads
    
    print(f"Test dimensions: batch={batch_size}, seq={seq_len}, hidden={hidden_size}")
    print(f"Heads: {num_heads}, KV heads: {num_kv_heads}, head_dim: {head_dim}")
    
    # Generate test weights (small matrices for easier debugging)
    np.random.seed(42)  # Reproducible
    q_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.01
    k_weight = np.random.randn(num_kv_heads * head_dim, hidden_size).astype(np.float32) * 0.01
    v_weight = np.random.randn(num_kv_heads * head_dim, hidden_size).astype(np.float32) * 0.01
    o_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.01
    
    test_input = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32) * 0.1
    
    print(f"Weight shapes: Q{q_weight.shape}, K{k_weight.shape}, V{v_weight.shape}, O{o_weight.shape}")
    
    # PyTorch computation
    try:
        import torch
        
        def pytorch_attention(x, q_w, k_w, v_w, o_w):
            # Projections (PyTorch uses input @ weight.T)
            q = torch.matmul(x, torch.from_numpy(q_w).T)
            k = torch.matmul(x, torch.from_numpy(k_w).T)
            v = torch.matmul(x, torch.from_numpy(v_w).T)
            
            # Reshape for attention
            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            
            # GQA expansion
            k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
            v = v.repeat_interleave(num_heads // num_kv_heads, dim=1)
            
            # Attention computation
            scale = 1.0 / (head_dim ** 0.5)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
            
            # Reshape and project
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
            output = torch.matmul(attn_output, torch.from_numpy(o_w).T)
            
            return output.numpy()
        
        pt_input = torch.from_numpy(test_input)
        pt_result = pytorch_attention(pt_input, q_weight, k_weight, v_weight, o_weight)
        
        print(f"PyTorch result shape: {pt_result.shape}")
        
    except Exception as e:
        print(f"PyTorch attention failed: {e}")
        return False
    
    # JAX computation
    try:
        import jax.numpy as jnp
        
        def jax_attention(x, q_w, k_w, v_w, o_w):
            # Projections (JAX uses input @ weight)
            q = jnp.dot(x, q_w.T)  # Transpose since weights are in PyTorch format
            k = jnp.dot(x, k_w.T)
            v = jnp.dot(x, v_w.T)
            
            # Reshape for attention
            q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
            
            # GQA expansion
            k = jnp.repeat(k, num_heads // num_kv_heads, axis=1)
            v = jnp.repeat(v, num_heads // num_kv_heads, axis=1)
            
            # Attention computation
            scale = 1.0 / (head_dim ** 0.5)
            attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
            attn_weights = jax.nn.softmax(attn_weights, axis=-1)
            attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)
            
            # Reshape and project
            attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_size)
            output = jnp.dot(attn_output, o_w.T)  # Transpose since weights are in PyTorch format
            
            return np.array(output)
        
        import jax
        jax_input = jnp.array(test_input)
        jax_result = jax_attention(jax_input, q_weight, k_weight, v_weight, o_weight)
        
        print(f"JAX result shape: {jax_result.shape}")
        
    except Exception as e:
        print(f"JAX attention failed: {e}")
        return False
    
    # Compare
    diff = np.abs(pt_result - jax_result).max()
    print(f"Small attention max difference: {diff:.8e}")
    
    if diff < 1e-6:
        print("✅ Small attention computation matches perfectly")
        return True
    else:
        print("❌ Small attention computation differs")
        print(f"Sample PyTorch: {pt_result.flat[:5]}")
        print(f"Sample JAX: {jax_result.flat[:5]}")
        print(f"Sample diff: {(pt_result - jax_result).flat[:5]}")
        return False

def main():
    """Run final precision tests to identify remaining issues"""
    print("FINAL PRECISION FIX")
    print("Goal: Identify remaining 0.1-3 logits differences to achieve 1e-6 tolerance")
    
    # Test 1: RMS norm precision
    rmsnorm_ok = check_rmsnorm_implementation()
    
    # Test 2: Dtype precision
    dtype_ok = check_dtype_precision()
    
    # Test 3: Small attention computation
    attention_ok = test_small_attention_computation()
    
    if rmsnorm_ok and dtype_ok and attention_ok:
        print("\n🎉 ALL PRECISION TESTS PASSED!")
        print("📋 Issue likely in model-specific implementation details")
        return True
    else:
        print("\n❌ Some precision tests failed")
        print("📋 Found specific areas for final fixes")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 