#!/usr/bin/env python3
"""
Corrected test with proper mesh setup
"""
import os
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

# Import and setup Qwen's global mesh properly
import q25j7_tensor_parallel_fixed
from q25j7_tensor_parallel_fixed import ParallelDense as QwenParallelDense

class LlamaParallelDense(nn.Module):
    """Exact copy of Llama's ParallelDense implementation"""
    features: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        in_dim = x.shape[-1]
        out_dim = self.features
        local_shape = (in_dim, out_dim)

        kernel = self.param(
            "kernel", nn.initializers.lecun_normal(), local_shape, self.param_dtype
        )

        def matmul_fn(x, k):
            axis_idx = jax.lax.axis_index("mp")
            local_out = jnp.einsum("bsd,df->bsf", x, k)
            full_out = jax.lax.all_gather(local_out, axis_name="mp", axis=0)
            return jnp.reshape(
                jnp.transpose(full_out, (1, 2, 0, 3)), (x.shape[0], x.shape[1], -1)
            )

        return shard_map(
            matmul_fn,
            mesh=mesh,  # Use global mesh
            in_specs=(None, P(None, "mp")),
            out_specs=P(None),
            check_rep=False,
        )(x, kernel)

def test_corrected_comparison():
    """Test with properly corrected mesh setup"""
    
    # Setup global mesh variables for both implementations
    devices = jax.devices()
    global mesh
    mesh = Mesh(devices, axis_names=("mp",))
    
    # Also set the mesh in the Qwen module
    q25j7_tensor_parallel_fixed.mesh = mesh
    
    print("=== CORRECTED LLAMA vs QWEN COMPARISON ===\n")
    print(f"Global mesh: {mesh}")
    print(f"Qwen module mesh: {q25j7_tensor_parallel_fixed.mesh}")
    
    # Test parameters
    input_shape = (2, 10, 64)
    output_features = 128
    
    # Create test input (same for both)
    rng = jax.random.PRNGKey(0)
    input_data = jnp.ones(input_shape, dtype=jnp.float32)
    
    print(f"Input shape: {input_shape}")
    print(f"Output features: {output_features}")
    
    # Test 1: Exact Llama implementation
    print("\n--- EXACT LLAMA IMPLEMENTATION ---")
    llama_layer = LlamaParallelDense(
        features=output_features,
        dtype=jnp.float32,
        param_dtype=jnp.float32
    )
    
    llama_params = llama_layer.init(rng, input_data)
    print(f"Llama kernel shape: {llama_params['params']['kernel'].shape}")
    
    with mesh:
        llama_output = llama_layer.apply(llama_params, input_data)
    
    print(f"Llama output shape: {llama_output.shape}")
    print(f"Llama output min/max: {float(jnp.min(llama_output)):.4f}, {float(jnp.max(llama_output)):.4f}")
    
    # Test 2: Current Qwen implementation (now with proper mesh)
    print("\n--- QWEN IMPLEMENTATION (WITH PROPER MESH) ---")
    qwen_layer = QwenParallelDense(
        features=output_features,
        dtype=jnp.float32,
        param_dtype=jnp.float32
    )
    
    # Use same RNG for fair comparison
    qwen_params = qwen_layer.init(rng, input_data)
    print(f"Qwen params structure: {list(qwen_params['params'].keys())}")
    
    # Should now have 'kernel' instead of 'Dense_0'
    if 'kernel' in qwen_params['params']:
        print(f"Qwen kernel shape: {qwen_params['params']['kernel'].shape}")
        print("✅ Qwen is using ParallelDense (has 'kernel')")
    else:
        print(f"❌ Qwen is falling back to nn.Dense (has {list(qwen_params['params'].keys())})")
    
    with mesh:
        qwen_output = qwen_layer.apply(qwen_params, input_data)
    
    print(f"Qwen output shape: {qwen_output.shape}")
    print(f"Qwen output min/max: {float(jnp.min(qwen_output)):.4f}, {float(jnp.max(qwen_output)):.4f}")
    
    # Compare
    print("\n--- COMPARISON ---")
    if llama_output.shape == qwen_output.shape:
        print("✅ Shapes match!")
        
        # Check if outputs are identical (they should be with same RNG)
        if jnp.allclose(llama_output, qwen_output, rtol=1e-5):
            print("✅ Outputs are identical!")
        else:
            max_diff = float(jnp.max(jnp.abs(llama_output - qwen_output)))
            print(f"❌ Outputs differ! Max difference: {max_diff:.6f}")
    else:
        print(f"❌ Shape mismatch: Llama {llama_output.shape} vs Qwen {qwen_output.shape}")
    
    return llama_output, qwen_output

if __name__ == "__main__":
    test_corrected_comparison()