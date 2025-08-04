#!/usr/bin/env python3
"""
Test model with ONLY O projection parallelized (which we know works)
"""
import os
import jax
import jax.numpy as jnp
import json
import flax.linen as nn
from typing import Dict, Any

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import q25j7_tensor_parallel_fixed
from q25j7_tensor_parallel_fixed import (
    ParallelDense, ParallelEmbed, QwenMLP, 
    compute_cos_sin_cache, apply_rotary_emb, make_causal_mask,
    setup_device_mesh, load_params
)
from transformers import AutoTokenizer

class OOnlyParallelAttention(nn.Module):
    """Attention with only O projection parallelized"""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        self.num_heads = c["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = c.get("num_key_value_heads", self.num_heads)
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.rope_theta = c.get("rope_theta", 1000000.0)
        
        # Use regular Dense for Q, K, V (like working hybrid model)
        self.q_proj = nn.Dense(self.hidden_size, dtype=jnp.bfloat16, name="q_proj")
        self.k_proj = nn.Dense(self.kv_dim, dtype=jnp.bfloat16, name="k_proj")
        self.v_proj = nn.Dense(self.kv_dim, dtype=jnp.bfloat16, name="v_proj")
        
        # Use ParallelDense ONLY for O projection (which we know works)
        self.o_proj = ParallelDense(
            self.hidden_size, 
            dtype=jnp.bfloat16, 
            param_dtype=jnp.bfloat16, 
            use_bias=False, 
            name="o_proj"
        )

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        batch, seq, _ = hidden_states.shape

        # Project inputs using regular Dense for Q, K, V
        q = self.q_proj(hidden_states).reshape(batch, seq, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(batch, seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(batch, seq, self.num_kv_heads, self.head_dim)

        # Apply rotary embeddings
        if position_ids is not None:
            cos, sin = compute_cos_sin_cache(position_ids, self.head_dim, self.rope_theta)
            q, k = apply_rotary_emb(q, k, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = jnp.concatenate([past_k, k], axis=1)
            v = jnp.concatenate([past_v, v], axis=1)

        cache_k, cache_v = k, v

        # GQA: Repeat k/v to match query heads
        if self.num_heads != self.num_kv_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = jnp.repeat(k, repeat, axis=2)
            v = jnp.repeat(v, repeat, axis=2)

        # Attention computation
        q = q.transpose(0, 2, 1, 3)  # [batch, heads, seq, head_dim]
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
        if attention_mask is not None:
            scores += attention_mask
        scores = scores.astype(jnp.float64)
        probs = jnp.clip(jax.nn.softmax(scores.astype(jnp.float32), axis=-1), 1e-9, 1 - 1e-9)
        attn_out = jnp.einsum('bhqk,bhkd->bhqd', probs, v)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq, self.hidden_size)

        # Use ParallelDense for output projection (which we know works)
        return self.o_proj(attn_out), (cache_k, cache_v)

class OOnlyParallelDecoderLayer(nn.Module):
    """Decoder with O-only parallel attention"""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        self.input_layernorm = nn.RMSNorm(epsilon=c.get("rms_norm_eps", 1e-6), dtype=jnp.bfloat16, name="input_layernorm")
        self.self_attn = OOnlyParallelAttention(config=c, dtype=jnp.bfloat16)
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

class OOnlyParallelModel(nn.Module):
    """Model with O-only parallel attention + parallel MLP + parallel LM head"""
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
        key_len = seq if past_key_values is None or past_key_values[0] is None else past_key_values[0][0].shape[1] + seq

        if attention_mask is None:
            attention_mask = jnp.ones((batch, 1, seq, key_len), dtype=self.dtype)
        causal_mask = make_causal_mask(seq, key_len)[None, None, :, :]
        attention_bias = jnp.where(attention_mask == 0, -1e9, 0) + causal_mask

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
            
            hidden_states, new_kv = layer(hidden_states, attention_bias, position_ids, past_kv)
            new_key_values.append(new_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if return_dict:
            return {"logits": logits, "past_key_values": new_key_values}
        return (logits,)

def test_o_only_parallel():
    """Test O-only parallel model"""
    
    # Setup
    mesh = setup_device_mesh()
    q25j7_tensor_parallel_fixed.mesh = mesh
    
    with open("weights/config.json") as f:
        config = json.load(f)
    
    print("=== TESTING O-ONLY PARALLEL MODEL ===\n")
    print("This should work perfectly since we know O projection parallelization works")
    
    # Create model
    model = OOnlyParallelModel(config=config, dtype=jnp.bfloat16)
    params = load_params(model, "weights", jnp.bfloat16)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("weights")
    
    # Test with same input
    test_input = "What is 2 + 2?"
    input_ids = tokenizer.encode(test_input, return_tensors="jax")
    
    print(f"Input: '{test_input}'")
    print(f"Input IDs: {input_ids}")
    
    # Test the model
    with mesh:
        outputs = model.apply(params, input_ids, return_dict=True)
        logits = outputs['logits'][0, -1, :]
    
    print(f"Logits min/max: {float(jnp.min(logits)):.4f}, {float(jnp.max(logits)):.4f}")
    print(f"Logits mean/std: {float(jnp.mean(logits)):.4f}, {float(jnp.std(logits)):.4f}")
    
    # Get top tokens
    top_tokens = jnp.argsort(logits)[-10:][::-1]
    top_probs = jax.nn.softmax(logits)[top_tokens]
    
    print("\nTop 10 predicted tokens:")
    for i, (token_id, prob) in enumerate(zip(top_tokens, top_probs)):
        token_text = tokenizer.decode(int(token_id))
        print(f"  {i+1}. Token {token_id}: '{token_text}' (prob: {float(prob):.4f})")
    
    # Check if this matches the original working model
    next_token = int(jnp.argmax(logits))
    next_token_text = tokenizer.decode(next_token)
    
    print(f"\nPredicted next token: {next_token} -> '{next_token_text}'")
    
    # This should predict space like the working models
    if next_token == 220:  # Token 220 is ' ' (space)
        print("✅ PERFECT! O-only parallel model works like original!")
        print("This confirms that the issue is in Q/K/V parallelization, not O parallelization")
        return True
    else:
        print(f"❌ O-only parallel model still differs from original")
        print("This suggests the issue is elsewhere")
        return False

if __name__ == "__main__":
    success = test_o_only_parallel()
    
    if success:
        print("\n🎯 CONFIRMED: The issue is specifically in Q/K/V projection parallelization!")
        print("Next step: Figure out what's different about input projection parallelization")
    else:
        print("\n🤔 The issue might be more fundamental than just Q/K/V parallelization")