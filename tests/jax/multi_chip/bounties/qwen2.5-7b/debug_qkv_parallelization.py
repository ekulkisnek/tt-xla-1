#!/usr/bin/env python3
"""
Debug script to isolate Q/K/V parallelization issues
"""

import jax
import jax.numpy as jnp
from jax.experimental import shard_map
from jax.sharding import Mesh, PartitionSpec as P
import flax.linen as nn
import os

# Setup mesh
devices = jax.devices()
mesh = Mesh(devices, ('mp',))
print(f'Using mesh: {mesh}')

# Test 1: Regular Dense (baseline)
print('\n=== Test 1: Regular Dense (baseline) ===')
class RegularDense(nn.Module):
    features: int
    def setup(self):
        self.dense = nn.Dense(self.features, dtype=jnp.bfloat16)
    def __call__(self, x):
        return self.dense(x)

# Test 2: ParallelDense (problematic)
print('\n=== Test 2: ParallelDense (problematic) ===')
class ParallelDense(nn.Module):
    features: int
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    use_bias: bool = False
    name: str = None

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            'kernel',
            nn.initializers.normal(stddev=0.02),
            (x.shape[-1], self.features),
            self.param_dtype,
        )
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.features,), self.param_dtype)
        else:
            bias = None

        def matmul_fn(x, k):
            axis_idx = jax.lax.axis_index('mp')
            local_out = jnp.einsum('bsd,df->bsf', x, k)
            full_out = jax.lax.all_gather(local_out, axis_name='mp', axis=0)
            return jnp.reshape(jnp.transpose(full_out, (1, 2, 0, 3)), (x.shape[0], x.shape[1], -1))

        return shard_map(
            matmul_fn,
            mesh=mesh,
            in_specs=(None, P(None, 'mp')),
            out_specs=P(None),
            check_rep=False,
        )(x, kernel)

# Test 3: Compare outputs
print('\n=== Test 3: Compare outputs ===')
batch_size, seq_len, hidden_size = 1, 8, 3584
test_input = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.bfloat16)

# Initialize models
regular_model = RegularDense(3584)
parallel_model = ParallelDense(3584)

# Get parameters
regular_params = regular_model.init(jax.random.PRNGKey(0), test_input)
parallel_params = parallel_model.init(jax.random.PRNGKey(0), test_input)

print(f'Regular params keys: {list(regular_params["params"].keys())}')
print(f'Parallel params keys: {list(parallel_params["params"].keys())}')

# Test forward pass
regular_output = regular_model.apply(regular_params, test_input)
print(f'Regular output shape: {regular_output.shape}')
print(f'Regular output dtype: {regular_output.dtype}')

parallel_output = parallel_model.apply(parallel_params, test_input)
print(f'Parallel output shape: {parallel_output.shape}')
print(f'Parallel output dtype: {parallel_output.dtype}')

# Check if outputs are similar
print(f'Outputs similar: {jnp.allclose(regular_output, parallel_output, atol=1e-3)}')

# Test 4: Test with real attention dimensions
print('\n=== Test 4: Test with real attention dimensions ===')
num_heads = 32
head_dim = 112
num_kv_heads = 4
kv_dim = num_kv_heads * head_dim

# Test Q projection (full hidden_size)
q_regular = RegularDense(3584)
q_parallel = ParallelDense(3584)

q_regular_params = q_regular.init(jax.random.PRNGKey(0), test_input)
q_parallel_params = q_parallel.init(jax.random.PRNGKey(0), test_input)

q_regular_out = q_regular.apply(q_regular_params, test_input)
q_parallel_out = q_parallel.apply(q_parallel_params, test_input)

print(f'Q projection outputs similar: {jnp.allclose(q_regular_out, q_parallel_out, atol=1e-3)}')

# Test K/V projection (kv_dim)
k_regular = RegularDense(kv_dim)
k_parallel = ParallelDense(kv_dim)

k_regular_params = k_regular.init(jax.random.PRNGKey(0), test_input)
k_parallel_params = k_parallel.init(jax.random.PRNGKey(0), test_input)

k_regular_out = k_regular.apply(k_regular_params, test_input)
k_parallel_out = k_parallel.apply(k_parallel_params, test_input)

print(f'K projection outputs similar: {jnp.allclose(k_regular_out, k_parallel_out, atol=1e-3)}')

print('\n=== Test 5: Test head reshaping ===')
# Reshape outputs to attention heads
q_regular_reshaped = q_regular_out.reshape(batch_size, seq_len, num_heads, head_dim)
q_parallel_reshaped = q_parallel_out.reshape(batch_size, seq_len, num_heads, head_dim)

k_regular_reshaped = k_regular_out.reshape(batch_size, seq_len, num_kv_heads, head_dim)
k_parallel_reshaped = k_parallel_out.reshape(batch_size, seq_len, num_kv_heads, head_dim)

print(f'Q reshaped shapes: {q_regular_reshaped.shape} vs {q_parallel_reshaped.shape}')
print(f'K reshaped shapes: {k_regular_reshaped.shape} vs {k_parallel_reshaped.shape}')

print(f'Q reshaped similar: {jnp.allclose(q_regular_reshaped, q_parallel_reshaped, atol=1e-3)}')
print(f'K reshaped similar: {jnp.allclose(k_regular_reshaped, k_parallel_reshaped, atol=1e-3)}')

print('\n=== Test 6: Test GQA repeat_kv ===')
# Test GQA repeat_kv
def repeat_kv(hidden_states: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :]
    hidden_states = jnp.repeat(hidden_states, n_rep, axis=3)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)

num_key_value_groups = num_heads // num_kv_heads
k_regular_repeated = repeat_kv(k_regular_reshaped, num_key_value_groups)
k_parallel_repeated = repeat_kv(k_parallel_reshaped, num_key_value_groups)

print(f'K repeated shapes: {k_regular_repeated.shape} vs {k_parallel_repeated.shape}')
print(f'K repeated similar: {jnp.allclose(k_regular_repeated, k_parallel_repeated, atol=1e-3)}')

print('\n=== Test 7: Test attention computation ===')
# Test attention computation
q_regular_transposed = q_regular_reshaped.transpose(0, 2, 1, 3)  # [batch, heads, seq, head_dim]
k_regular_transposed = k_regular_repeated.transpose(0, 2, 1, 3)

q_parallel_transposed = q_parallel_reshaped.transpose(0, 2, 1, 3)
k_parallel_transposed = k_parallel_repeated.transpose(0, 2, 1, 3)

# Compute attention scores
scale = 1.0 / jnp.sqrt(head_dim)
scores_regular = jnp.einsum('bhqd,bhkd->bhqk', q_regular_transposed, k_regular_transposed) * scale
scores_parallel = jnp.einsum('bhqd,bhkd->bhqk', q_parallel_transposed, k_parallel_transposed) * scale

print(f'Attention scores shapes: {scores_regular.shape} vs {scores_parallel.shape}')
print(f'Attention scores similar: {jnp.allclose(scores_regular, scores_parallel, atol=1e-3)}')

print('\n=== SUMMARY ===')
print('If all tests pass, the issue is NOT in the individual components.')
print('The problem must be in the model-level integration or weight loading.') 