#!/usr/bin/env python3
"""
Test script for ParallelDense implementation
"""
import os
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

# Global mesh
mesh = None

class ParallelDense(nn.Module):
    """Tensor parallel dense layer that shards weights across devices."""
    features: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    use_bias: bool = False
    name: str = None

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        in_dim = x.shape[-1]
        out_dim = self.features
        
        # If we only have 1 device, fall back to regular Dense
        if mesh is None or mesh.shape["mp"] == 1:
            # Use regular Dense for single device
            return nn.Dense(
                self.features, 
                dtype=self.dtype, 
                param_dtype=self.param_dtype,
                use_bias=self.use_bias,
                name=self.name
            )(x)
        
        local_shape = (in_dim, out_dim)

        # Use the same parameter name structure as regular Dense layers
        kernel = self.param(
            "kernel", nn.initializers.lecun_normal(), local_shape, self.param_dtype
        )

        def matmul_fn(x, k):
            axis_idx = jax.lax.axis_index("mp")
            print(f"🔧 Device {axis_idx} running matmul: x.shape = {x.shape}, kernel.shape = {k.shape}")

            local_out = jnp.einsum("bsd,df->bsf", x, k)
            print(f"🔧 Device {axis_idx} local_out shape: {local_out.shape}")

            full_out = jax.lax.all_gather(local_out, axis_name="mp", axis=0)
            print(f"🔧 Device {axis_idx} full_out shape: {full_out.shape}")

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

def test_parallel_dense():
    """Test ParallelDense with a simple input"""
    global mesh
    
    # Setup mesh
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=("mp",))
    print(f"Created mesh: {mesh}")
    
    # Create model
    model = ParallelDense(features=128, dtype=jnp.float32, name="test_dense")
    
    # Create test input
    batch_size, seq_len, hidden_size = 2, 10, 64
    x = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)
    
    print(f"Input shape: {x.shape}")
    
    # Initialize parameters
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, x)
    
    print("Parameters initialized successfully")
    
    # Test forward pass
    with mesh:
        output = model.apply(params, x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: {x.shape[:-1] + (128,)}")
    
    assert output.shape == (batch_size, seq_len, 128), f"Expected {(batch_size, seq_len, 128)}, got {output.shape}"
    print("✅ ParallelDense test passed!")

if __name__ == "__main__":
    test_parallel_dense() 