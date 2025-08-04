#!/usr/bin/env python3
"""
Test to compare exact Llama vs current Qwen ParallelDense implementations
"""
import os
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

from q25j7_tensor_parallel_fixed import ParallelDense as QwenParallelDense

# Global mesh variable (like in Llama)
mesh = None

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
            mesh=mesh,
            in_specs=(None, P(None, "mp")),
            out_specs=P(None),
            check_rep=False,
        )(x, kernel)

def test_both_implementations():
    """Test both implementations side by side"""
    
    global mesh
    
    # Setup mesh
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=("mp",))
    
    print("=== COMPARING LLAMA vs QWEN PARALLEL DENSE ===\n")
    
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
    
    # Test 2: Current Qwen implementation
    print("\n--- CURRENT QWEN IMPLEMENTATION ---")
    qwen_layer = QwenParallelDense(
        features=output_features,
        dtype=jnp.float32,
        param_dtype=jnp.float32
    )
    
    # Use same RNG for fair comparison
    qwen_params = qwen_layer.init(rng, input_data)
    print(f"Qwen params structure: {list(qwen_params.keys())}")
    if 'params' in qwen_params:
        print(f"Qwen params['params'] keys: {list(qwen_params['params'].keys())}")
        if 'kernel' in qwen_params['params']:
            print(f"Qwen kernel shape: {qwen_params['params']['kernel'].shape}")
        else:
            print("No 'kernel' in qwen_params['params']")
    else:
        print("No 'params' in qwen_params")
    
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
            
            # Check if kernels are identical
            llama_kernel = llama_params['params']['kernel']
            qwen_kernel = qwen_params['params']['kernel']
            
            if jnp.allclose(llama_kernel, qwen_kernel):
                print("✅ Kernels are identical, difference is in computation")
            else:
                print("❌ Kernels differ")
    else:
        print(f"❌ Shape mismatch: Llama {llama_output.shape} vs Qwen {qwen_output.shape}")
    
    return llama_output, qwen_output

if __name__ == "__main__":
    test_both_implementations()