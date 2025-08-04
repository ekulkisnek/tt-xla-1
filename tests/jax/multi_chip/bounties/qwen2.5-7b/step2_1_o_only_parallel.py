#!/usr/bin/env python3
"""
Step 2.1: Test Llama attention with only O-projection parallelized (gradual approach)
"""
import os
import jax
import jax.numpy as jnp
import json
import flax.linen as nn
from typing import Dict, Any, Optional, Union
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from jax import lax

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import q25j7_tensor_parallel_fixed
from q25j7_tensor_parallel_fixed import (
    ParallelDense, ParallelEmbed, QwenMLP, 
    setup_device_mesh, load_params
)
from transformers import AutoTokenizer

# Import exact functions from Step 1
def compute_cos_sin_cache(position_ids, head_dim, rope_theta=1000000.0):
    """Qwen's working rotary embedding computation."""
    pos = position_ids.astype(jnp.float32)  # [batch, seq]
    dim = head_dim // 2
    freqs = 1.0 / (rope_theta ** (jnp.arange(0, dim, dtype=jnp.float32) / dim))
    t = pos[..., None] * freqs[None, None, :]
    cos = jnp.cos(t)
    sin = jnp.sin(t)
    # Expand for broadcasting: [batch, seq, 1, dim]
    cos = cos[..., None, :]
    sin = sin[..., None, :]
    return cos, sin

def apply_rotary_emb(q, k, cos, sin):
    """Qwen's working rotary embedding application."""
    # q, k: [batch, seq, heads, head_dim]
    # cos, sin: [batch, seq, 1, dim] where dim = head_dim // 2
    half_dim = q.shape[-1] // 2
    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    k1, k2 = k[..., :half_dim], k[..., half_dim:]
    # cos and sin are already [batch, seq, 1, dim], so they broadcast correctly
    q_rot = jnp.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
    k_rot = jnp.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)
    return q_rot, k_rot

def repeat_kv(hidden_states: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    """Repeat key/value states for GQA (exact Llama implementation)."""
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :]
    hidden_states = jnp.repeat(hidden_states, n_rep, axis=3)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)

# Create a mock config class like Llama's
class MockLlamaConfig:
    def __init__(self, qwen_config):
        # Map Qwen config to Llama-style config
        self.hidden_size = qwen_config["hidden_size"]
        self.num_attention_heads = qwen_config["num_attention_heads"] 
        self.num_key_value_heads = qwen_config.get("num_key_value_heads", self.num_attention_heads)
        self.max_sequence_length = qwen_config.get("max_position_embeddings", 32768)
        self.rope_theta = qwen_config.get("rope_theta", 10000.0)
        self.attn_pdrop = 0.0  # Llama default

class OOnlyParallelAttention(nn.Module):
    """Llama attention with ONLY O-projection parallelized (gradual approach)"""
    config: MockLlamaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # Use REGULAR Dense for Q/K/V projections (known to work)
        self.q_proj = nn.Dense(
            config.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,  # Qwen Q projection has bias
        )

        self.k_proj = nn.Dense(
            config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,  # Qwen K projection has bias
        )

        self.v_proj = nn.Dense(
            config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,  # Qwen V projection has bias
        )

        # Use ParallelDense ONLY for O projection (known to work)
        self.o_proj = ParallelDense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,  # Qwen O projection has no bias
        )

        self.causal_mask = make_causal_mask(
            jnp.ones((1, config.max_sequence_length), dtype="bool"), dtype="bool"
        )
        
        # Store config for rotary embedding computation
        self.rope_theta = config.rope_theta

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(
            hidden_states.shape[:2] + (num_heads, self.head_dim)
        )

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """EXACT copy of Llama's cache logic"""
        # detect if we're initializing by absence of existing cache data.
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable(
            "cache", "cached_key", jnp.zeros, key.shape, key.dtype
        )
        cached_value = self.variable(
            "cache", "cached_value", jnp.zeros, value.shape, value.dtype
        )
        cache_index = self.variable(
            "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)
        )

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        # EXACT copy of Llama's attention computation, but Q/K/V use regular Dense
        if attention_mask is None:
            attention_mask = jnp.ones((hidden_states.shape[0], hidden_states.shape[1]), dtype=jnp.float32)
        
        # Q/K/V use regular Dense (no parallelism) 
        xq, xk, xv = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

        xq = self._split_heads(xq, self.num_heads)
        xk = self._split_heads(xk, self.num_key_value_heads)
        xv = self._split_heads(xv, self.num_key_value_heads)

        # Use Qwen-style rotary embedding computation
        cos, sin = compute_cos_sin_cache(position_ids, self.head_dim, self.rope_theta)
        xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        query_length, key_length = xq.shape[1], xk.shape[1]

        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                self.causal_mask,
                (0, 0, mask_shift, 0),
                (1, 1, query_length, max_decoder_length),
            )
        else:
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(
            causal_mask, (batch_size,) + causal_mask.shape[1:]
        )

        attention_mask = jnp.broadcast_to(
            jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape
        )
        attention_mask = combine_masks(attention_mask, causal_mask)

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.has_variable("cache", "cached_key") or init_cache:
            xk, xv, attention_mask = self._concatenate_to_cache(
                xk, xv, xq, attention_mask
            )

        # transform boolean mask into float mask
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(
                self.dtype
            ),
        )

        xk = repeat_kv(xk, self.num_key_value_groups)
        xv = repeat_kv(xv, self.num_key_value_groups)

        # usual dot product attention
        attn_weights = dot_product_attention_weights(
            xq,
            xk,
            bias=attention_bias,
            dropout_rng=None,
            dropout_rate=self.config.attn_pdrop,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=self.precision,
        )

        attn_output = jnp.einsum(
            "...hqk,...khd->...qhd", attn_weights, xv, precision=self.precision
        )
        attn_output = self._merge_heads(attn_output)
        
        # O projection uses ParallelDense (this works!)
        attn_output = self.o_proj(attn_output)

        # Return format compatible with Qwen interface
        cache_k, cache_v = xk, xv  # Simplified cache for now
        return attn_output, (cache_k, cache_v)

class OOnlyParallelDecoderLayer(nn.Module):
    """Decoder layer using O-only parallel attention"""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        llama_config = MockLlamaConfig(c)
        
        self.input_layernorm = nn.RMSNorm(epsilon=c.get("rms_norm_eps", 1e-6), dtype=jnp.bfloat16, name="input_layernorm")
        self.self_attn = OOnlyParallelAttention(config=llama_config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
        self.post_attention_layernorm = nn.RMSNorm(epsilon=c.get("rms_norm_eps", 1e-6), dtype=jnp.bfloat16, name="post_attention_layernorm")
        self.mlp = QwenMLP(config=c, dtype=jnp.bfloat16)

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, past_key_value = self.self_attn(hidden_states, attention_mask, position_ids, past_key_value)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, past_key_value

class OOnlyParallelQwenModel(nn.Module):
    """Full Qwen model with O-only parallel attention"""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        self.embed_tokens = ParallelEmbed(c["vocab_size"], c["hidden_size"], dtype=jnp.bfloat16, name="embed_tokens")
        self.layers = [OOnlyParallelDecoderLayer(config=c, dtype=jnp.bfloat16, name=f"layers_{i}") for i in range(c["num_hidden_layers"])]
        self.norm = nn.RMSNorm(epsilon=c.get("rms_norm_eps", 1e-6), dtype=jnp.bfloat16, name="norm")
        self.lm_head = ParallelDense(c["vocab_size"], dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name="lm_head")

    def __call__(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, return_dict=True):
        batch, seq = input_ids.shape
        
        # Use same attention mask logic as original
        if attention_mask is None:
            attention_mask = jnp.ones((batch, seq), dtype=jnp.float32)

        hidden_states = self.embed_tokens(input_ids)
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        new_key_values = []
        
        for layer, past_kv in zip(self.layers, past_key_values):
            if position_ids is None:
                if past_kv is None:
                    position_ids = jnp.arange(seq)[None, :].repeat(batch, axis=0)
                else:
                    start_pos = past_kv[0].shape[1]
                    position_ids = jnp.arange(start_pos, start_pos + seq)[None, :].repeat(batch, axis=0)
            
            hidden_states, new_kv = layer(hidden_states, attention_mask, position_ids, past_kv)
            new_key_values.append(new_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if return_dict:
            return {"logits": logits, "past_key_values": new_key_values}
        return (logits,)

def test_o_only_parallel_model():
    """Test Step 2.1: Llama attention with only O-projection parallelized"""
    
    # Setup
    mesh = setup_device_mesh()
    q25j7_tensor_parallel_fixed.mesh = mesh
    
    with open("weights/config.json") as f:
        config = json.load(f)
    
    print("=== STEP 2.1: TESTING O-ONLY PARALLEL LLAMA ATTENTION ===\n")
    print("Goal: Test Llama attention flow with only O-projection parallelized")
    print("Strategy: Q/K/V use regular Dense (known to work), O uses ParallelDense")
    
    # Create model
    model = OOnlyParallelQwenModel(config=config, dtype=jnp.bfloat16)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("weights")
    
    # Test with same input as always
    test_input = "What is 2 + 2?"
    input_ids = tokenizer.encode(test_input, return_tensors="jax")
    
    print(f"Input: '{test_input}'")
    print(f"Input IDs: {input_ids}")
    
    # Load weights
    print("\n--- LOADING WEIGHTS ---")
    try:
        params = load_params(model, "weights", jnp.bfloat16)
        print("✅ Weight loading successful")
        
    except Exception as e:
        print(f"❌ Weight loading failed: {e}")
        return False
    
    # Test the model
    print("\n--- TESTING MODEL ---")
    try:
        with mesh:
            outputs = model.apply(params, input_ids, return_dict=True)
            logits = outputs['logits'][0, -1, :]
        
        print(f"✅ Forward pass successful")
        print(f"Logits shape: {logits.shape}")
        print(f"Logits min/max: {float(jnp.min(logits)):.4f}, {float(jnp.max(logits)):.4f}")
        print(f"Logits mean/std: {float(jnp.mean(logits)):.4f}, {float(jnp.std(logits)):.4f}")
        
        # Get top tokens
        top_tokens = jnp.argsort(logits)[-10:][::-1]
        top_probs = jax.nn.softmax(logits)[top_tokens]
        
        print("\nTop 10 predicted tokens:")
        for i, (token_id, prob) in enumerate(zip(top_tokens, top_probs)):
            token_text = tokenizer.decode(int(token_id))
            print(f"  {i+1}. Token {token_id}: '{token_text}' (prob: {float(prob):.4f})")
        
        # Check prediction
        next_token = int(jnp.argmax(logits))
        next_token_text = tokenizer.decode(next_token)
        
        print(f"\nPredicted next token: {next_token} -> '{next_token_text}'")
        
        # Check if this matches the original working model (space token)
        if next_token == 220:  # Token 220 is ' ' (space)
            print("🎉 STEP 2.1 SUCCESS: O-only parallel Llama attention produces CORRECT output!")
            print("This proves Llama attention flow is compatible with Qwen!")
            print("The issue is specifically with parallelizing Q/K/V projections.")
            return True
        else:
            print(f"❌ Step 2.1 failed: Output differs from expected")
            print(f"Expected: 220 (' '), Got: {next_token} ('{next_token_text}')")
            return False
        
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = test_o_only_parallel_model()
    
    if result:
        print("\n✅ STEP 2.1 COMPLETE: Llama attention flow is proven compatible!")
        print("Next: Investigate why Q/K/V parallelization breaks the flow")
    else:
        print("\n❌ STEP 2.1 FAILED: Need to debug the hybrid approach")