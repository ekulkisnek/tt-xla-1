#!/usr/bin/env python3
"""
Test parallelizing attention projections one at a time to isolate the issue
"""
import os
import jax
import jax.numpy as jnp
import json
import flax.linen as nn
from typing import Dict, Any

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

# Import components
import q25j7_tensor_parallel_fixed
from q25j7_tensor_parallel_fixed import (
    ParallelDense, ParallelEmbed, QwenMLP, 
    compute_cos_sin_cache, apply_rotary_emb, make_causal_mask,
    setup_device_mesh, load_params
)
from transformers import AutoTokenizer

class IncrementalAttention(nn.Module):
    """Test attention with one projection at a time parallelized"""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    parallel_layers: tuple = ()  # Which layers to parallelize: ('q', 'k', 'v', 'o')

    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        self.num_heads = c["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = c.get("num_key_value_heads", self.num_heads)
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.rope_theta = c.get("rope_theta", 1000000.0)
        
        # Choose ParallelDense or nn.Dense based on parallel_layers
        if 'q' in self.parallel_layers:
            self.q_proj = ParallelDense(self.hidden_size, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name="q_proj")
        else:
            self.q_proj = nn.Dense(self.hidden_size, dtype=jnp.bfloat16, name="q_proj")
            
        if 'k' in self.parallel_layers:
            self.k_proj = ParallelDense(self.kv_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name="k_proj")
        else:
            self.k_proj = nn.Dense(self.kv_dim, dtype=jnp.bfloat16, name="k_proj")
            
        if 'v' in self.parallel_layers:
            self.v_proj = ParallelDense(self.kv_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name="v_proj")
        else:
            self.v_proj = nn.Dense(self.kv_dim, dtype=jnp.bfloat16, name="v_proj")
            
        if 'o' in self.parallel_layers:
            self.o_proj = ParallelDense(self.hidden_size, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, use_bias=False, name="o_proj")
        else:
            self.o_proj = nn.Dense(self.hidden_size, dtype=jnp.bfloat16, use_bias=False, name="o_proj")

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        batch, seq, _ = hidden_states.shape

        # Project inputs
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

        return self.o_proj(attn_out), (cache_k, cache_v)

def create_test_model(parallel_layers):
    """Create a test model with specific layers parallelized"""
    
    class TestQwenDecoderLayer(nn.Module):
        config: Dict[str, Any]
        dtype: jnp.dtype = jnp.float32

        def setup(self):
            c = self.config
            self.input_layernorm = nn.RMSNorm(epsilon=c.get("rms_norm_eps", 1e-6), dtype=jnp.bfloat16, name="input_layernorm")
            self.self_attn = IncrementalAttention(config=c, dtype=jnp.bfloat16, parallel_layers=parallel_layers)
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

    class TestQwen25ForCausalLM(nn.Module):
        config: Dict[str, Any]
        dtype: jnp.dtype = jnp.float32

        def setup(self):
            c = self.config
            self.embed_tokens = ParallelEmbed(c["vocab_size"], c["hidden_size"], dtype=jnp.bfloat16, name="embed_tokens")
            self.layers = [TestQwenDecoderLayer(config=c, dtype=jnp.bfloat16, name=f"layers_{i}") for i in range(c["num_hidden_layers"])]
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
    
    return TestQwen25ForCausalLM

def test_incremental_attention():
    """Test parallelizing attention projections incrementally"""
    
    # Setup
    mesh = setup_device_mesh()
    q25j7_tensor_parallel_fixed.mesh = mesh
    
    with open("weights/config.json") as f:
        config = json.load(f)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("weights")
    
    # Test input
    test_input = "What is 2 + 2?"
    input_ids = tokenizer.encode(test_input, return_tensors="jax")
    
    print("=== INCREMENTAL ATTENTION DEBUGGING ===\n")
    print(f"Input: '{test_input}'")
    print(f"Goal: Find which projection causes the problem\n")
    
    # Test different combinations
    test_cases = [
        ((), "Baseline (all regular Dense)"),
        (('q',), "Only Q parallel"),
        (('k',), "Only K parallel"), 
        (('v',), "Only V parallel"),
        (('o',), "Only O parallel"),
        (('q', 'k'), "Q+K parallel"),
        (('q', 'v'), "Q+V parallel"),
        (('q', 'o'), "Q+O parallel"),
        (('k', 'v'), "K+V parallel"),
        (('k', 'o'), "K+O parallel"),
        (('v', 'o'), "V+O parallel"),
        (('q', 'k', 'v'), "Q+K+V parallel"),
        (('q', 'k', 'o'), "Q+K+O parallel"),
        (('q', 'v', 'o'), "Q+V+O parallel"),
        (('k', 'v', 'o'), "K+V+O parallel"),
        (('q', 'k', 'v', 'o'), "All parallel (broken)"),
    ]
    
    results = []
    
    for parallel_layers, description in test_cases:
        print(f"--- {description} ---")
        
        try:
            # Create model with specific parallel layers
            ModelClass = create_test_model(parallel_layers)
            model = ModelClass(config=config, dtype=jnp.bfloat16)
            params = load_params(model, "weights", jnp.bfloat16)
            
            # Test the model
            with mesh:
                outputs = model.apply(params, input_ids, return_dict=True)
                logits = outputs['logits'][0, -1, :]
            
            # Get top token
            top_token = int(jnp.argmax(logits))
            top_token_text = tokenizer.decode(top_token)
            top_prob = float(jax.nn.softmax(logits)[top_token])
            
            print(f"Top token: {top_token} -> '{top_token_text}' (prob: {top_prob:.4f})")
            
            # Check if it matches the expected space token
            if top_token == 220:  # Space token
                print("✅ CORRECT - matches original model")
                status = "✅ CORRECT"
            else:
                print("❌ WRONG - different from original")
                status = "❌ WRONG"
            
            results.append((parallel_layers, description, top_token, top_token_text, status))
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            results.append((parallel_layers, description, None, None, "❌ FAILED"))
        
        print()
    
    # Summary
    print("=== SUMMARY ===")
    for parallel_layers, description, top_token, top_token_text, status in results:
        print(f"{status} {description}: {top_token} -> '{top_token_text}'")
    
    # Find the transition point
    print("\n=== ANALYSIS ===")
    correct_cases = [r for r in results if r[4] == "✅ CORRECT"]
    wrong_cases = [r for r in results if r[4] == "❌ WRONG"]
    
    if correct_cases and wrong_cases:
        print(f"✅ {len(correct_cases)} cases work correctly")
        print(f"❌ {len(wrong_cases)} cases have issues")
        print("\nThe problem likely starts when adding one of these layers:")
        for case in wrong_cases:
            if len(case[0]) == 1:  # Single layer cases
                print(f"  - {case[0][0]} projection")
    else:
        print("Need more investigation")

if __name__ == "__main__":
    test_incremental_attention()