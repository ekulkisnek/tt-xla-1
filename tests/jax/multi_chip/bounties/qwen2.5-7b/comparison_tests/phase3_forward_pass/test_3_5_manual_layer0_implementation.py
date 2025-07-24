#!/usr/bin/env python3
"""
Phase 3.5: Manual Layer 0 Implementation
========================================
Implements step-by-step JAX Layer 0 computation and compares each component
with PyTorch to identify exactly where divergence occurs.
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
    """Get test input for layer 0 analysis"""
    return "Hello"  # Token 9707 - confirmed identical embedding

def extract_pytorch_layer0_components(test_input):
    """Extract detailed Layer 0 components from PyTorch model"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print("Loading PyTorch model for detailed Layer 0 analysis...")
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
        
        # Extract embedding (confirmed identical)
        with torch.no_grad():
            embedding_output = model.model.embed_tokens(input_ids)
            
            # Get Layer 0 components
            layer_0 = model.model.layers[0]
            
            # Step 1: Input layer norm
            hidden_states = embedding_output
            normalized_input = layer_0.input_layernorm(hidden_states)
            
            # Step 2: Self-attention
            attn_output, attn_weights, _ = layer_0.self_attn(
                normalized_input,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=True
            )
            
            # Step 3: Residual connection after attention
            hidden_states_after_attn = hidden_states + attn_output
            
            # Step 4: Post-attention layer norm
            normalized_after_attn = layer_0.post_attention_layernorm(hidden_states_after_attn)
            
            # Step 5: MLP
            mlp_output = layer_0.mlp(normalized_after_attn)
            
            # Step 6: Final residual connection
            layer_0_final_output = hidden_states_after_attn + mlp_output
            
        results = {
            'embedding_output': embedding_output.float().numpy(),
            'normalized_input': normalized_input.float().numpy(),
            'attn_output': attn_output.float().numpy(),
            'attn_weights': attn_weights.float().numpy(),
            'hidden_after_attn': hidden_states_after_attn.float().numpy(),
            'normalized_after_attn': normalized_after_attn.float().numpy(),
            'mlp_output': mlp_output.float().numpy(),
            'layer_0_final': layer_0_final_output.float().numpy(),
            'tokens': input_ids.numpy()
        }
        
        print(f"PyTorch Layer 0 components extracted:")
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

def manual_rms_norm_jax(x, weight, eps=1e-6):
    """Manual RMS normalization for JAX matching PyTorch's exact implementation"""
    import jax.numpy as jnp
    
    # Match PyTorch's exact implementation:
    # 1. Save input dtype
    input_dtype = x.dtype
    # 2. Convert to float32 for computation
    hidden_states = x.astype(jnp.float32)
    # 3. Compute variance
    variance = jnp.mean(hidden_states**2, axis=-1, keepdims=True)
    # 4. Apply rsqrt (reciprocal square root) like PyTorch
    hidden_states = hidden_states * jnp.power(variance + eps, -0.5)  # rsqrt equivalent
    # 5. Convert back to input dtype
    hidden_states = hidden_states.astype(input_dtype)
    # 6. Apply weight
    return weight * hidden_states

def manual_rope_jax(q, k, position_ids, rope_theta=10000.0):
    """Manual RoPE implementation for JAX"""
    import jax.numpy as jnp
    
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Create frequency tensor
    freqs = 1.0 / (rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    
    # Create position embeddings
    position_ids = position_ids.astype(jnp.float32)
    freqs = jnp.outer(position_ids.flatten(), freqs)
    
    # Create rotation matrix
    cos_freqs = jnp.cos(freqs)
    sin_freqs = jnp.sin(freqs)
    
    # Reshape for broadcasting
    cos_freqs = cos_freqs.reshape(batch_size, seq_len, 1, head_dim // 2)
    sin_freqs = sin_freqs.reshape(batch_size, seq_len, 1, head_dim // 2)
    
    # Split q and k into even/odd indices
    q_even = q[..., ::2]
    q_odd = q[..., 1::2]
    k_even = k[..., ::2] 
    k_odd = k[..., 1::2]
    
    # Apply rotation
    q_rotated_even = q_even * cos_freqs - q_odd * sin_freqs
    q_rotated_odd = q_even * sin_freqs + q_odd * cos_freqs
    
    k_rotated_even = k_even * cos_freqs - k_odd * sin_freqs
    k_rotated_odd = k_even * sin_freqs + k_odd * cos_freqs
    
    # Recombine
    q_rotated = jnp.stack([q_rotated_even, q_rotated_odd], axis=-1).reshape(q.shape)
    k_rotated = jnp.stack([k_rotated_even, k_rotated_odd], axis=-1).reshape(k.shape)
    
    return q_rotated, k_rotated

def extract_jax_layer0_components(test_input):
    """Manually implement JAX Layer 0 computation step-by-step"""
    try:
        import jax
        import jax.numpy as jnp
        sys.path.append("../..")
        
        from qwen_jax_inference import Qwen25ForCausalLM, load_params
        from transformers import AutoTokenizer
        
        print("Loading JAX model for manual Layer 0 implementation...")
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
        
        # Get embedding (confirmed identical)
        embedding_weights = params['params']['embed_tokens']['embedding']
        token_id = input_ids[0, 0]
        embedding_output = embedding_weights[token_id][None, None, :]  # [1, 1, hidden_size]
        
        # Get Layer 0 parameters
        layer_0_params = params['params']['layers_0']
        
        # Extract all parameter components
        input_ln_scale = layer_0_params['input_layernorm']['scale']
        attn_q_kernel = layer_0_params['self_attn']['q_proj']['kernel']
        attn_q_bias = layer_0_params['self_attn']['q_proj']['bias']
        attn_k_kernel = layer_0_params['self_attn']['k_proj']['kernel']
        attn_k_bias = layer_0_params['self_attn']['k_proj']['bias']
        attn_v_kernel = layer_0_params['self_attn']['v_proj']['kernel']
        attn_v_bias = layer_0_params['self_attn']['v_proj']['bias']
        attn_o_kernel = layer_0_params['self_attn']['o_proj']['kernel']
        attn_o_bias = layer_0_params['self_attn']['o_proj'].get('bias', None)  # o_proj has no bias
        
        post_ln_scale = layer_0_params['post_attention_layernorm']['scale']
        
        mlp_gate_kernel = layer_0_params['mlp']['gate_proj']['kernel']
        mlp_up_kernel = layer_0_params['mlp']['up_proj']['kernel']
        mlp_down_kernel = layer_0_params['mlp']['down_proj']['kernel']
        
        # Manual computation step-by-step
        hidden_states = embedding_output.astype(jnp.bfloat16)  # Match PyTorch precision
        
        # Step 1: Input layer normalization (in bfloat16)
        normalized_input = manual_rms_norm_jax(hidden_states, input_ln_scale.astype(jnp.bfloat16))
        
        print(f"Step 1 - Input normalization: {normalized_input.shape}")
        
        # Step 2: Self-attention
        batch_size, seq_len, hidden_size = normalized_input.shape
        num_heads = config['num_attention_heads']
        num_key_value_heads = config.get('num_key_value_heads', num_heads)
        head_dim = hidden_size // num_heads
        
        # Q, K, V projections (in bfloat16)
        q = jnp.dot(normalized_input, attn_q_kernel.astype(jnp.bfloat16)) + attn_q_bias.astype(jnp.bfloat16)
        k = jnp.dot(normalized_input, attn_k_kernel.astype(jnp.bfloat16)) + attn_k_bias.astype(jnp.bfloat16)
        v = jnp.dot(normalized_input, attn_v_kernel.astype(jnp.bfloat16)) + attn_v_bias.astype(jnp.bfloat16)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, num_heads, head_dim)
        k = k.reshape(batch_size, seq_len, num_key_value_heads, head_dim)
        v = v.reshape(batch_size, seq_len, num_key_value_heads, head_dim)
        
        print(f"Step 2a - Q/K/V shapes: {q.shape}, {k.shape}, {v.shape}")
        
        # Apply RoPE (keep in bfloat16)
        position_ids = jnp.arange(seq_len).reshape(1, seq_len)
        q_rope, k_rope = manual_rope_jax(q, k, position_ids)
        
        print(f"Step 2b - After RoPE: {q_rope.shape}, {k_rope.shape}")
        
        # Attention computation (use mixed precision like PyTorch)
        # Transpose for attention: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        q_t = jnp.transpose(q_rope, (0, 2, 1, 3))
        k_t = jnp.transpose(k_rope, (0, 2, 1, 3))
        v_t = jnp.transpose(v, (0, 2, 1, 3))
        
        # Handle Grouped Query Attention (GQA): expand K and V to match Q heads
        if num_key_value_heads != num_heads:
            # Calculate how many times to repeat each K/V head
            heads_per_group = num_heads // num_key_value_heads
            
            # Expand K and V: [batch, kv_heads, seq, head_dim] -> [batch, q_heads, seq, head_dim]
            k_t = jnp.repeat(k_t, heads_per_group, axis=1)
            v_t = jnp.repeat(v_t, heads_per_group, axis=1)
            
            print(f"Step 2b.1 - GQA expansion: K {k_t.shape}, V {v_t.shape}")
        
        # Attention scores (mixed precision like PyTorch softmax)
        # Convert to float32 for precision-sensitive computation
        q_t_f32 = q_t.astype(jnp.float32)
        k_t_f32 = k_t.astype(jnp.float32)
        v_t_f32 = v_t.astype(jnp.float32)
        
        attn_scores = jnp.matmul(q_t_f32, jnp.transpose(k_t_f32, (0, 1, 3, 2))) / math.sqrt(head_dim)
        
        # Causal mask for single token (not needed for seq_len=1, but let's be explicit)
        if seq_len > 1:
            mask = jnp.tril(jnp.ones((seq_len, seq_len)))
            attn_scores = jnp.where(mask, attn_scores, -jnp.inf)
        
        # Softmax in float32 (like PyTorch)
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        
        # Apply attention (keep in float32)
        attn_output = jnp.matmul(attn_weights, v_t_f32)
        
        # Convert back to original dtype before transpose/reshape
        attn_output = attn_output.astype(q_rope.dtype)
        
        # Transpose back and reshape
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)
        
        # Output projection (in bfloat16)
        attn_output = jnp.dot(attn_output, attn_o_kernel.astype(jnp.bfloat16))
        if attn_o_bias is not None:
            attn_output = attn_output + attn_o_bias.astype(jnp.bfloat16)
        
        print(f"Step 2c - Attention output: {attn_output.shape}")
        
        # Step 3: Residual connection after attention
        hidden_after_attn = hidden_states + attn_output
        
        print(f"Step 3 - After attention residual: {hidden_after_attn.shape}")
        
        # Step 4: Post-attention layer normalization (in bfloat16)
        normalized_after_attn = manual_rms_norm_jax(hidden_after_attn, post_ln_scale.astype(jnp.bfloat16))
        
        print(f"Step 4 - Post-attention normalization: {normalized_after_attn.shape}")
        
        # Step 5: MLP (in bfloat16)
        # Gate and up projections
        gate = jnp.dot(normalized_after_attn, mlp_gate_kernel.astype(jnp.bfloat16))
        up = jnp.dot(normalized_after_attn, mlp_up_kernel.astype(jnp.bfloat16))
        
        # SiLU activation on gate
        gate_activated = gate * jax.nn.sigmoid(gate)
        
        # Element-wise multiply
        mlp_intermediate = gate_activated * up
        
        # Down projection
        mlp_output = jnp.dot(mlp_intermediate, mlp_down_kernel.astype(jnp.bfloat16))
        
        print(f"Step 5 - MLP output: {mlp_output.shape}")
        
        # Step 6: Final residual connection
        layer_0_final = hidden_after_attn + mlp_output
        
        print(f"Step 6 - Layer 0 final: {layer_0_final.shape}")
        
        # Convert to float32 only for final comparison
        results = {
            'embedding_output': np.array(embedding_output.astype(jnp.float32), dtype=np.float32),
            'normalized_input': np.array(normalized_input.astype(jnp.float32), dtype=np.float32),
            'attn_output': np.array(attn_output.astype(jnp.float32), dtype=np.float32),
            'attn_weights': np.array(attn_weights.astype(jnp.float32), dtype=np.float32),
            'hidden_after_attn': np.array(hidden_after_attn.astype(jnp.float32), dtype=np.float32),
            'normalized_after_attn': np.array(normalized_after_attn.astype(jnp.float32), dtype=np.float32),
            'mlp_output': np.array(mlp_output.astype(jnp.float32), dtype=np.float32),
            'layer_0_final': np.array(layer_0_final.astype(jnp.float32), dtype=np.float32),
            'tokens': input_ids
        }
        
        print(f"JAX Layer 0 manual computation completed:")
        for key, value in results.items():
            if key != 'tokens':
                print(f"  {key}: shape {value.shape}")
        
        # Cleanup
        del model, params
        jax.clear_caches()
        gc.collect()
        
        return results, None
        
    except Exception as e:
        return None, str(e)

def compare_layer0_components(pytorch_results, jax_results, tolerance=1e-4):
    """Compare Layer 0 components step by step"""
    
    print("\n" + "=" * 70)
    print("LAYER 0 COMPONENT-BY-COMPONENT ANALYSIS")
    print("=" * 70)
    
    components = [
        'embedding_output',
        'normalized_input', 
        'attn_output',
        'attn_weights',
        'hidden_after_attn',
        'normalized_after_attn',
        'mlp_output',
        'layer_0_final'
    ]
    
    results = []
    first_divergence = None
    
    for component in components:
        if component in pytorch_results and component in jax_results:
            pt_data = pytorch_results[component]
            jax_data = jax_results[component]
            
            print(f"\n--- {component.upper()} ---")
            print(f"PyTorch shape: {pt_data.shape}")
            print(f"JAX shape: {jax_data.shape}")
            
            if pt_data.shape != jax_data.shape:
                print(f"❌ Shape mismatch!")
                results.append((component, False, "Shape mismatch", 0))
                if first_divergence is None:
                    first_divergence = component
                continue
            
            # Compare values
            diff = np.abs(pt_data - jax_data)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"Max difference: {max_diff:.2e}")
            print(f"Mean difference: {mean_diff:.2e}")
            print(f"Tolerance: {tolerance:.2e}")
            
            within_tolerance = max_diff < tolerance
            
            if within_tolerance:
                print(f"✅ {component} MATCHES")
                status = "MATCH"
            else:
                print(f"❌ {component} DIFFERS")
                status = "DIFFER"
                
                # Show detailed stats for differing components
                print(f"  95th percentile: {np.percentile(diff, 95):.2e}")
                print(f"  99th percentile: {np.percentile(diff, 99):.2e}")
                
                if first_divergence is None:
                    first_divergence = component
            
            results.append((component, within_tolerance, status, max_diff))
    
    # Summary analysis
    print("\n" + "=" * 70)
    print("DIVERGENCE ANALYSIS")
    print("=" * 70)
    
    print(f"\nComponent-by-component results:")
    for component, matches, status, max_diff in results:
        status_symbol = "✅" if matches else "❌"
        print(f"  {component:<20} {status_symbol} {status:<8} max_diff: {max_diff:.2e}")
    
    if first_divergence:
        print(f"\n🎯 FIRST DIVERGENCE: {first_divergence}")
        print(f"All components before {first_divergence} match within tolerance")
        print(f"Issue identified in {first_divergence} computation")
    else:
        print(f"\n✅ ALL COMPONENTS MATCH!")
        print(f"Layer 0 computation is identical between PyTorch and JAX")
    
    return results, first_divergence

def main():
    """Main test function"""
    print("=" * 70)
    print("PHASE 3.5: MANUAL LAYER 0 IMPLEMENTATION")
    print("=" * 70)
    
    test_input = get_test_input()
    print(f"Test input: '{test_input}'")
    
    # Extract PyTorch Layer 0 components
    print(f"\n{'='*50}")
    print("EXTRACTING PYTORCH LAYER 0 COMPONENTS")
    print(f"{'='*50}")
    
    pytorch_results, pytorch_error = extract_pytorch_layer0_components(test_input)
    if pytorch_error:
        print(f"❌ PyTorch extraction failed: {pytorch_error}")
        return False
    
    print("✅ PyTorch Layer 0 components extracted")
    
    # Extract JAX Layer 0 components manually
    print(f"\n{'='*50}")
    print("MANUAL JAX LAYER 0 COMPUTATION")
    print(f"{'='*50}")
    
    jax_results, jax_error = extract_jax_layer0_components(test_input)
    if jax_error:
        print(f"❌ JAX manual computation failed: {jax_error}")
        return False
    
    print("✅ JAX Layer 0 manual computation completed")
    
    # Compare components
    results, first_divergence = compare_layer0_components(pytorch_results, jax_results)
    
    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 3.5 SUMMARY")
    print("=" * 70)
    
    matching_components = sum(1 for _, matches, _, _ in results if matches)
    total_components = len(results)
    
    print(f"Components analyzed: {total_components}")
    print(f"Matching components: {matching_components}")
    print(f"Differing components: {total_components - matching_components}")
    
    if first_divergence:
        print(f"\n🎯 CRITICAL FINDING:")
        print(f"First divergence in: {first_divergence}")
        print(f"Focus debugging efforts on {first_divergence} implementation")
        return False
    else:
        print(f"\n✅ LAYER 0 COMPUTATION MATCHES!")
        print(f"Issue must be in later layers or final processing")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 