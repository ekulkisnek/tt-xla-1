#!/usr/bin/env python3
"""
Direct comparison between Llama's ParallelDense and our implementation
"""

import jax
import jax.numpy as jnp
from jax.experimental import shard_map
from jax.sharding import Mesh, PartitionSpec as P
import flax.linen as nn

# Setup mesh
devices = jax.devices()
mesh = Mesh(devices, ('mp',))
print(f'Using mesh: {mesh}')

# Llama's exact ParallelDense implementation
class LlamaParallelDense(nn.Module):
    features: float
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        in_dim = x.shape[-1]
        out_dim = self.features
        local_shape = (in_dim, out_dim)  # Full output dimension

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

# Our ParallelDense implementation
class OurParallelDense(nn.Module):
    features: int
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    use_bias: bool = False
    name: str = None

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        in_dim = x.shape[-1]
        out_dim = self.features
        
        # If we only have 1 device, fall back to regular Dense
        if mesh is None or mesh.shape["mp"] == 1:
            return nn.Dense(
                self.features, 
                dtype=self.dtype, 
                param_dtype=self.param_dtype,
                use_bias=self.use_bias,
                name=self.name
            )(x)
        
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

print('\n=== Testing with single device (should use fallback) ===')
batch_size, seq_len, hidden_size = 1, 8, 3584
test_input = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.bfloat16)

# Test our implementation
our_model = OurParallelDense(3584)
our_params = our_model.init(jax.random.PRNGKey(0), test_input)
our_output = our_model.apply(our_params, test_input)
print(f'Our output shape: {our_output.shape}')

# Test Llama implementation
llama_model = LlamaParallelDense(3584)
llama_params = llama_model.init(jax.random.PRNGKey(0), test_input)
llama_output = llama_model.apply(llama_params, test_input)
print(f'Llama output shape: {llama_output.shape}')

print(f'Outputs similar: {jnp.allclose(our_output, llama_output, atol=1e-3)}')

print('\n=== Testing with multiple devices ===')
# Force multiple devices by creating a larger mesh
if len(devices) > 1:
    print('Multiple devices available - testing parallel execution')
    # Test with real attention dimensions
    num_heads = 32
    head_dim = 112
    num_kv_heads = 4
    kv_dim = num_kv_heads * head_dim
    
    # Test Q projection
    our_q = OurParallelDense(3584)
    llama_q = LlamaParallelDense(3584)
    
    our_q_params = our_q.init(jax.random.PRNGKey(0), test_input)
    llama_q_params = llama_q.init(jax.random.PRNGKey(0), test_input)
    
    our_q_out = our_q.apply(our_q_params, test_input)
    llama_q_out = llama_q.apply(llama_q_params, test_input)
    
    print(f'Q projection outputs similar: {jnp.allclose(our_q_out, llama_q_out, atol=1e-3)}')
    
    # Test K projection
    our_k = OurParallelDense(kv_dim)
    llama_k = LlamaParallelDense(kv_dim)
    
    our_k_params = our_k.init(jax.random.PRNGKey(0), test_input)
    llama_k_params = llama_k.init(jax.random.PRNGKey(0), test_input)
    
    our_k_out = our_k.apply(our_k_params, test_input)
    llama_k_out = llama_k.apply(llama_k_params, test_input)
    
    print(f'K projection outputs similar: {jnp.allclose(our_k_out, llama_k_out, atol=1e-3)}')
else:
    print('Only single device available - cannot test parallel execution')

print('\n=== Key differences identified ===')
print('1. Llama uses float32, we use bfloat16')
print('2. Llama has no fallback for single device')
print('3. Llama uses lecun_normal initialization')
print('4. Our implementation has use_bias parameter')

print('\n=== Hypothesis ===')
print('The issue might be the fallback to regular Dense in single device mode.')
print('This could cause parameter loading mismatches when the model expects ParallelDense.') 