#!/usr/bin/env python3
"""
Debug the actual tensor sharding happening in ParallelDense
"""
import os
import jax
import jax.numpy as jnp
import flax.linen as nn

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import q25j7_tensor_parallel_fixed
from q25j7_tensor_parallel_fixed import ParallelDense, setup_device_mesh

def debug_actual_sharding():
    """Debug what shapes each device actually gets"""
    
    # Setup
    mesh = setup_device_mesh()
    q25j7_tensor_parallel_fixed.mesh = mesh
    
    print("=== DEBUGGING ACTUAL TENSOR SHARDING ===\n")
    
    # Test parameters
    input_dim = 3584
    output_dim = 512  # K/V dimension that fails
    
    print(f"Testing K/V projection: {input_dim} → {output_dim}")
    print(f"Expected local output per device: {output_dim // mesh.shape['mp']} = {output_dim // 4}")
    
    # Create layer
    layer = ParallelDense(output_dim, dtype=jnp.float32, param_dtype=jnp.float32)
    
    # Test input
    rng = jax.random.PRNGKey(42)
    input_data = jnp.ones((1, 1, input_dim), dtype=jnp.float32)
    
    # Initialize
    params = layer.init(rng, input_data)
    kernel = params['params']['kernel']
    
    print(f"Initialized kernel shape: {kernel.shape}")
    print(f"Expected kernel shape: ({input_dim}, {output_dim})")
    
    if kernel.shape != (input_dim, output_dim):
        print("❌ Kernel shape is wrong!")
        return False
    
    # Create a modified version of ParallelDense that shows device shapes
    class DebuggingParallelDense(nn.Module):
        features: int
        dtype: jnp.dtype = jnp.float32
        param_dtype: jnp.dtype = jnp.float32

        @nn.compact
        def __call__(self, x):
            x = x.astype(self.dtype)
            in_dim = x.shape[-1]
            out_dim = self.features
            local_shape = (in_dim, out_dim)

            kernel = self.param("kernel", nn.initializers.lecun_normal(), local_shape, self.param_dtype)

            def matmul_fn(x, k):
                axis_idx = jax.lax.axis_index("mp")
                print(f"Device {axis_idx}: x.shape = {x.shape}, k.shape = {k.shape}")
                
                local_out = jnp.einsum("bsd,df->bsf", x, k)
                print(f"Device {axis_idx}: local_out.shape = {local_out.shape}")
                
                full_out = jax.lax.all_gather(local_out, axis_name="mp", axis=0)
                print(f"Device {axis_idx}: full_out.shape after all_gather = {full_out.shape}")
                
                result = jnp.reshape(jnp.transpose(full_out, (1, 2, 0, 3)), (x.shape[0], x.shape[1], -1))
                print(f"Device {axis_idx}: final result.shape = {result.shape}")
                
                return result

            from jax.experimental.shard_map import shard_map
            from jax.sharding import PartitionSpec as P
            
            return shard_map(
                matmul_fn,
                mesh=mesh,
                in_specs=(None, P(None, "mp")),
                out_specs=P(None),
                check_rep=False,
            )(x, kernel)
    
    # Test with debugging version
    print("\n--- TESTING WITH DEBUGGING OUTPUT ---")
    debug_layer = DebuggingParallelDense(output_dim, dtype=jnp.float32, param_dtype=jnp.float32)
    debug_params = debug_layer.init(rng, input_data)
    debug_params['params']['kernel'] = kernel  # Use same kernel
    
    try:
        with mesh:
            output = debug_layer.apply(debug_params, input_data)
        
        print(f"\nFinal output shape: {output.shape}")
        print(f"Expected output shape: (1, 1, {output_dim})")
        
        if output.shape == (1, 1, output_dim):
            print("✅ Output shape is correct")
            return True
        else:
            print("❌ Output shape is wrong")
            return False
            
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_actual_sharding()