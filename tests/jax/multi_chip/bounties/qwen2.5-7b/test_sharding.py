#!/usr/bin/env python3
"""
Test to verify that ParallelDense is actually sharding computation.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
import time

# Global mesh
mesh = None

class TestParallelDense:
    """Test version of ParallelDense with debug prints."""
    
    def __init__(self, features, mesh):
        self.features = features
        self.mesh = mesh
    
    def __call__(self, x):
        in_dim = x.shape[-1]
        out_dim = self.features
        
        # Create dummy parameters
        kernel = jnp.ones((in_dim, out_dim), dtype=jnp.bfloat16)
        bias = jnp.zeros((out_dim,), dtype=jnp.bfloat16)
        
        def matmul_fn(x, k, b):
            axis_idx = jax.lax.axis_index("mp")
            print(f"🔧 Device {axis_idx}: Processing shard of shape {k.shape}")
            local_out = jnp.einsum("bsd,df->bsf", x, k)
            print(f"🔧 Device {axis_idx}: Local output shape {local_out.shape}")
            
            if b is not None:
                local_out = local_out + b
            
            full_out = jax.lax.all_gather(local_out, axis_name="mp", axis=0)
            print(f"🔧 Device {axis_idx}: Gathered output shape {full_out.shape}")
            
            result = jnp.reshape(
                jnp.transpose(full_out, (1, 2, 0, 3)), (x.shape[0], x.shape[1], -1)
            )
            print(f"🔧 Device {axis_idx}: Final result shape {result.shape}")
            return result
        
        output = shard_map(
            matmul_fn,
            mesh=self.mesh,
            in_specs=(None, P(None, "mp"), P("mp",)),
            out_specs=P(None),
            check_rep=False,
        )(x, kernel, bias)
        
        return output

def test_sharding():
    """Test that sharding is actually happening."""
    global mesh
    
    print("=== TESTING SHARDING ===")
    
    # Setup mesh
    devices = jax.devices()
    print(f"Available devices: {len(devices)}")
    mesh = Mesh(devices, axis_names=("mp",))
    
    # Create test input
    batch_size, seq_len, hidden_size = 1, 8, 3584
    x = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.bfloat16)
    
    print(f"Input shape: {x.shape}")
    print(f"Expected output features: {hidden_size}")
    print(f"Number of devices: {len(devices)}")
    print(f"Expected shard size: {hidden_size // len(devices)}")
    
    # Test ParallelDense
    parallel_dense = TestParallelDense(hidden_size, mesh)
    
    print("\n--- Running ParallelDense ---")
    start_time = time.time()
    output = parallel_dense(x)
    end_time = time.time()
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    
    # Verify output is correct
    expected_shape = (batch_size, seq_len, hidden_size)
    if output.shape == expected_shape:
        print("✅ Output shape is correct")
    else:
        print(f"❌ Expected {expected_shape}, got {output.shape}")
    
    print("=== SHARDING TEST COMPLETE ===")

if __name__ == "__main__":
    test_sharding() 