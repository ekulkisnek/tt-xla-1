#!/usr/bin/env python3
"""
Test with exact copy of Llama's ParallelDense implementation
"""
import os
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

from q25j7_tensor_parallel_fixed import setup_device_mesh

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
            # Skip debug print for cleaner output
            # print(f"🔧 Device {axis_idx} running matmul: x.shape = {x.shape}, kernel.shape = {k.shape}")

            local_out = jnp.einsum("bsd,df->bsf", x, k)

            full_out = jax.lax.all_gather(local_out, axis_name="mp", axis=0)

            return jnp.reshape(
                jnp.transpose(full_out, (1, 2, 0, 3)), (x.shape[0], x.shape[1], -1)
            )

        # Note: we replicate x, shard only kernel
        return shard_map(
            matmul_fn,
            mesh=mesh,
            in_specs=(
                None,
                P(None, "mp"),
            ),  # x is replicated, kernel is sharded on output dim
            out_specs=P(None),  # output is sharded along output dim
            check_rep=False,
        )(x, kernel)

def test_exact_llama_implementation():
    """Test with exact Llama implementation"""
    
    global mesh
    
    # Setup mesh (like Llama)
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=("mp",))
    
    print("=== TESTING EXACT LLAMA PARALLEL DENSE ===\n")
    
    # Test parameters
    input_shape = (2, 10, 64)  # batch, seq, features
    output_features = 128
    
    # Create test input
    rng = jax.random.PRNGKey(0)
    input_data = jnp.ones(input_shape, dtype=jnp.float32)
    
    print(f"Input shape: {input_shape}")
    print(f"Output features: {output_features}")
    
    # Test Llama ParallelDense
    print("\n--- EXACT LLAMA PARALLEL DENSE ---")
    try:
        llama_layer = LlamaParallelDense(
            features=output_features,
            dtype=jnp.float32,
            param_dtype=jnp.float32
        )
        
        # Initialize
        llama_params = llama_layer.init(rng, input_data)
        print(f"Llama kernel shape: {llama_params['params']['kernel'].shape}")
        
        # Forward pass
        with mesh:
            llama_output = llama_layer.apply(llama_params, input_data)
        
        print(f"Llama output shape: {llama_output.shape}")
        print(f"Llama output min/max: {float(jnp.min(llama_output)):.4f}, {float(jnp.max(llama_output)):.4f}")
        print("✅ Exact Llama ParallelDense works!")
        
        return llama_output
        
    except Exception as e:
        print(f"❌ Exact Llama ParallelDense failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_exact_llama_implementation()