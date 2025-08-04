#!/usr/bin/env python3
"""
Debug exact differences between nn.Dense and ParallelDense outputs
"""
import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import json

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import q25j7_tensor_parallel_fixed
from q25j7_tensor_parallel_fixed import ParallelDense, setup_device_mesh, load_params
from transformers import AutoTokenizer

def debug_exact_projection_outputs():
    """Compare exact outputs of nn.Dense vs ParallelDense with real model weights"""
    
    # Setup
    mesh = setup_device_mesh()
    q25j7_tensor_parallel_fixed.mesh = mesh
    
    with open("weights/config.json") as f:
        config = json.load(f)
    
    print("=== DEBUGGING EXACT PROJECTION OUTPUTS ===\n")
    
    # Use REAL model input (not synthetic)
    tokenizer = AutoTokenizer.from_pretrained("weights")
    test_input = "What is 2 + 2?"
    input_ids = tokenizer.encode(test_input, return_tensors="jax")
    
    print(f"Using REAL input: '{test_input}'")
    print(f"Input IDs: {input_ids}")
    
    # Create a minimal model with just embedding + one layer to get real hidden states
    class MinimalModel(nn.Module):
        def setup(self):
            self.embed_tokens = nn.Embed(config["vocab_size"], config["hidden_size"], dtype=jnp.bfloat16, name="embed_tokens")
        
        def __call__(self, input_ids):
            return self.embed_tokens(input_ids)
    
    # Load real embedding weights
    model = MinimalModel()
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, input_ids)
    
    # Load real weights for embedding
    import safetensors
    weights_path = "weights/model.safetensors"
    with open(weights_path, "rb") as f:
        weights = safetensors.torch.load(f)
    
    # Load embedding weights
    embed_weight = weights["model.embed_tokens.weight"]
    embed_weight_jax = jnp.array(embed_weight).astype(jnp.bfloat16)
    params["params"]["embed_tokens"]["embedding"] = embed_weight_jax
    
    # Get real hidden states
    hidden_states = model.apply(params, input_ids)
    print(f"Real hidden states shape: {hidden_states.shape}")
    print(f"Hidden states range: {float(jnp.min(hidden_states)):.4f} to {float(jnp.max(hidden_states)):.4f}")
    
    # Focus on the LAST token (where prediction happens)
    last_hidden = hidden_states[0, -1:, :]  # Keep seq dimension for consistency
    print(f"Last token hidden state shape: {last_hidden.shape}")
    
    # Test Q projection specifically (since it failed in our incremental test)
    print("\n--- TESTING Q PROJECTION WITH REAL WEIGHTS ---")
    
    # Load real Q projection weights
    q_weight = weights["model.layers.0.self_attn.q_proj.weight"]
    q_weight_jax = jnp.array(q_weight).astype(jnp.bfloat16)
    
    print(f"Real Q weight shape: {q_weight_jax.shape}")
    
    # Test 1: Regular Dense
    print("\n1. Regular Dense Q projection:")
    q_dense = nn.Dense(config["hidden_size"], dtype=jnp.bfloat16, name="q_proj")
    q_dense_params = q_dense.init(rng, last_hidden)
    q_dense_params["params"]["kernel"] = q_weight_jax.T  # Transpose for Dense layer
    
    q_output_dense = q_dense.apply(q_dense_params, last_hidden)
    print(f"Dense Q output shape: {q_output_dense.shape}")
    print(f"Dense Q output range: {float(jnp.min(q_output_dense)):.4f} to {float(jnp.max(q_output_dense)):.4f}")
    print(f"Dense Q output mean: {float(jnp.mean(q_output_dense)):.4f}")
    print(f"Dense Q output std: {float(jnp.std(q_output_dense)):.4f}")
    
    # Test 2: ParallelDense
    print("\n2. ParallelDense Q projection:")
    q_parallel = ParallelDense(config["hidden_size"], dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name="q_proj")
    q_parallel_params = q_parallel.init(rng, last_hidden)
    q_parallel_params["params"]["kernel"] = q_weight_jax.T  # Same weights!
    
    with mesh:
        q_output_parallel = q_parallel.apply(q_parallel_params, last_hidden)
    
    print(f"Parallel Q output shape: {q_output_parallel.shape}")
    print(f"Parallel Q output range: {float(jnp.min(q_output_parallel)):.4f} to {float(jnp.max(q_output_parallel)):.4f}")
    print(f"Parallel Q output mean: {float(jnp.mean(q_output_parallel)):.4f}")
    print(f"Parallel Q output std: {float(jnp.std(q_output_parallel)):.4f}")
    
    # Compare outputs
    print("\n--- COMPARISON ---")
    if q_output_dense.shape == q_output_parallel.shape:
        print("✅ Shapes match")
        
        # Check if outputs are close
        max_diff = float(jnp.max(jnp.abs(q_output_dense - q_output_parallel)))
        mean_diff = float(jnp.mean(jnp.abs(q_output_dense - q_output_parallel)))
        rel_diff = max_diff / float(jnp.max(jnp.abs(q_output_dense))) if float(jnp.max(jnp.abs(q_output_dense))) > 0 else 0
        
        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        print(f"Relative difference: {rel_diff:.6f} ({rel_diff*100:.4f}%)")
        
        if max_diff < 1e-5:
            print("✅ Outputs are virtually identical!")
        elif max_diff < 1e-3:
            print("⚠️ Small differences (likely floating point)")
        else:
            print("❌ Significant differences!")
            
            # Show first few values for debugging
            print("\nFirst 10 values comparison:")
            print("Dense:   ", [f"{float(x):.4f}" for x in q_output_dense.flatten()[:10]])
            print("Parallel:", [f"{float(x):.4f}" for x in q_output_parallel.flatten()[:10]])
    else:
        print(f"❌ Shape mismatch: {q_output_dense.shape} vs {q_output_parallel.shape}")
    
    # If they're identical, the issue is elsewhere
    if max_diff < 1e-3:
        print("\n🔍 Q projection outputs are nearly identical!")
        print("The issue is likely in how multiple parallel projections interact,")
        print("not in individual ParallelDense implementations.")
        return True
    else:
        print("\n❌ Found significant difference in Q projection!")
        return False

if __name__ == "__main__":
    debug_exact_projection_outputs()