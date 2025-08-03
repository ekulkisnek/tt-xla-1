#!/usr/bin/env python3
"""
Optimization 5: Advanced Precomputed Rotary Embeddings with complex number approach
This implements the complex-valued frequency approach from the bounty implementations.
"""
import os
import sys
import json
import argparse
import logging
import psutil
import gc
import time
import jax.random
from typing import Dict, Any, Optional, Tuple

# Disable x64 globally for faster inference
os.environ["JAX_ENABLE_X64"] = "0"

# XLA flags for CPU optimization - enable multi-threading
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true"
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer
from flax import linen as nn
from jax.experimental import pjit

# Disable debug NaNs for speed
jax.config.update('jax_debug_nans', False)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("qwen25_opt5")

# --- OPTIMIZATION 5: Advanced Precomputed Rotary Embeddings ---
def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0):
    """Precompute rotary embedding frequencies using complex numbers.
    
    Args:
        dim: Head dimension (must be even)
        end: Maximum sequence length
        theta: RoPE theta parameter
    
    Returns:
        Complex-valued frequencies of shape [end, dim//2]
    """
    # Create frequency bands: [dim//2]
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)] / dim))
    
    # Create position indices: [end]
    t = jnp.arange(end)
    
    # Outer product: [end, dim//2]
    freqs = jnp.outer(t, freqs)
    
    # Convert to complex numbers: [end, dim//2]
    freqs_cis = jnp.exp(1j * freqs)
    
    return freqs_cis

def apply_rotary_emb_complex(xq: jnp.ndarray, xk: jnp.ndarray, freqs_cis: jnp.ndarray, dtype: jnp.dtype = jnp.float32):
    """Apply rotary embeddings using complex-valued frequencies."""
    # Reshape for complex multiplication
    xq_reshaped = xq.reshape(*xq.shape[:-1], -1, 2)
    xk_reshaped = xk.reshape(*xk.shape[:-1], -1, 2)
    
    # Convert to complex numbers
    xq_complex = xq_reshaped[..., 0] + 1j * xq_reshaped[..., 1]
    xk_complex = xk_reshaped[..., 0] + 1j * xk_reshaped[..., 1]
    
    # Apply complex multiplication
    freqs_cis_expanded = freqs_cis[None, :, None, :]  # [1, seq, 1, dim//2]
    xq_rot = xq_complex * freqs_cis_expanded
    xk_rot = xk_complex * freqs_cis_expanded
    
    # Convert back to real numbers
    xq_rot_real = jnp.stack([xq_rot.real, xq_rot.imag], axis=-1)
    xk_rot_real = jnp.stack([xk_rot.real, xk_rot.imag], axis=-1)
    
    # Reshape back to original shape
    xq_rot = xq_rot_real.reshape(*xq.shape)
    xk_rot = xk_rot_real.reshape(*xk.shape)
    
    return xq_rot.astype(dtype), xk_rot.astype(dtype)

# --- Model Code ---
class QwenAttention(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        self.num_heads = c["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = c.get("num_key_value_heads", self.num_heads)
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Dense(self.hidden_size, dtype=jnp.bfloat16, name="q_proj")
        self.k_proj = nn.Dense(self.kv_dim, dtype=jnp.bfloat16, name="k_proj")
        self.v_proj = nn.Dense(self.kv_dim, dtype=jnp.bfloat16, name="v_proj")
        self.o_proj = nn.Dense(self.hidden_size, dtype=jnp.bfloat16, use_bias=False, name="o_proj")
        self.rope_theta = c.get("rope_theta", 1000000.0)
        
        # OPTIMIZATION 5: Advanced precomputed rotary embeddings
        max_seq_len = c.get("max_sequence_length", 32768) * 2
        self.freqs_cis = precompute_freqs_cis(
            self.head_dim, max_seq_len, theta=self.rope_theta
        )

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        batch, seq, _ = hidden_states.shape

        # Project inputs
        q = self.q_proj(hidden_states).reshape(batch, seq, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(batch, seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(batch, seq, self.num_kv_heads, self.head_dim)

        # OPTIMIZATION 5: Apply rotary embeddings using complex approach
        if position_ids is not None:
            freqs_cis_lookup = jnp.take(self.freqs_cis, position_ids[0], axis=0)
            q, k = apply_rotary_emb_complex(q, k, freqs_cis_lookup, dtype=self.dtype)

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
        
        # Use higher precision for attention scores
        scores = scores.astype(jnp.float64)
        probs = jnp.clip(jax.nn.softmax(scores.astype(jnp.float32), axis=-1), 1e-9, 1 - 1e-9)
        attn_out = jnp.einsum('bhqk,bhkd->bhqd', probs, v)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq, self.hidden_size)

        return self.o_proj(attn_out), (cache_k, cache_v)

def make_causal_mask(q_len, k_len):
    i = jnp.arange(q_len)[:, None]
    j = jnp.arange(k_len)[None, :]
    return jnp.where(i >= j - (k_len - q_len), 0, -1e9)

class QwenMLP(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        self.intermediate_size = c.get("intermediate_size", 4 * c["hidden_size"])
        self.gate_proj = nn.Dense(self.intermediate_size, dtype=jnp.bfloat16, use_bias=False, name="gate_proj")
        self.up_proj = nn.Dense(self.intermediate_size, dtype=jnp.bfloat16, use_bias=False, name="up_proj")
        self.down_proj = nn.Dense(c["hidden_size"], dtype=jnp.bfloat16, use_bias=False, name="down_proj")

    def __call__(self, x):
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))

class QwenDecoderLayer(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        self.input_layernorm = nn.RMSNorm(epsilon=c.get("rms_norm_eps", 1e-6), dtype=jnp.bfloat16, name="input_layernorm")
        self.self_attn = QwenAttention(config=c, dtype=jnp.bfloat16)
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

class Qwen25ForCausalLM(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        self.embed_tokens = nn.Embed(c["vocab_size"], c["hidden_size"], dtype=jnp.bfloat16, name="embed_tokens")
        self.layers = [QwenDecoderLayer(config=c, dtype=jnp.bfloat16, name=f"layers_{i}") for i in range(c["num_hidden_layers"])]
        self.norm = nn.RMSNorm(epsilon=c.get("rms_norm_eps", 1e-6), dtype=jnp.bfloat16, name="norm")
        self.lm_head = nn.Dense(c["vocab_size"], dtype=jnp.bfloat16, use_bias=False, name="lm_head")

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
            hidden_states, new_kv = layer(hidden_states, attention_bias, position_ids, past_kv)
            new_key_values.append(new_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if return_dict:
            return {"logits": logits, "past_key_values": new_key_values}
        return logits

# --- Weight Loading ---
def get_param_path(name):
    mapping = {
        "model.embed_tokens.weight": ("embed_tokens", "embedding"),
        "model.norm.weight": ("norm", "scale"),
        "lm_head.weight": ("lm_head", "kernel"),
    }
    if name in mapping:
        return mapping[name]
    import re
    if m := re.match(r"model\.layers\.(\d+)\.(input|post_attention)_layernorm\.weight", name):
        return (f"layers_{m.group(1)}", f"{m.group(2)}_layernorm", "scale")
    if m := re.match(r"model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.(weight|bias)", name):
        return (f"layers_{m.group(1)}", "self_attn", f"{m.group(2)}_proj", "kernel" if m.group(3) == "weight" else "bias")
    if m := re.match(r"model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight", name):
        return (f"layers_{m.group(1)}", "mlp", f"{m.group(2)}_proj", "kernel")
    return None

def transpose_if_needed(name, param):
    if "weight" in name and "layernorm" not in name and "embed_tokens" not in name:
        return param.T
    return param

def load_params(model, model_path, dtype):
    """Load model parameters from safetensors files."""
    print(f"Loading JAX model weights from {model_path}...")
    params = {"params": {}}
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            with safe_open(os.path.join(model_path, file), framework="numpy") as f:
                for key in f.keys():
                    path = get_param_path(key)
                    if path:
                        param = f.get_tensor(key)
                        param = jnp.array(param, dtype=jnp.bfloat16)
                        param = transpose_if_needed(key, param)
                        d = params["params"]
                        for p in path[:-1]:
                            d = d.setdefault(p, {})
                        d[path[-1]] = param
    
    gc.collect()
    print("Weight loading completed")
    return params

# --- Generation ---
def sample_next_token(logits):
    """Simplified greedy sampling only."""
    return jnp.argmax(logits, axis=-1)[0]

def generate_text(model, jit_apply, params, tokenizer, max_tokens, input_ids, position_ids, verbose=False):
    batch, seq = input_ids.shape
    
    # Prefill KV cache for the prompt
    outputs = jit_apply(params, input_ids=input_ids, position_ids=position_ids, past_key_values=None, return_dict=True)
    logits = outputs["logits"]
    past_key_values = outputs["past_key_values"]
    next_token = sample_next_token(logits[:, -1, :])
    generated_tokens = input_ids[0].tolist() + [int(next_token)]
    input_ids = jnp.array([[next_token]])
    position_ids = position_ids[:, -1:] + 1

    start_time = time.perf_counter()
    peak_mem = psutil.virtual_memory().used / (1024**3)
    print(f"Initial memory before generation: {peak_mem:.2f} GB used")
    print(f"Free memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")

    num_tokens_generated = 1
    for i in range(1, max_tokens):
        outputs = jit_apply(params, input_ids=input_ids, position_ids=position_ids, past_key_values=past_key_values, return_dict=True)
        logits = outputs["logits"]
        past_key_values = outputs["past_key_values"]
        next_token = sample_next_token(logits[:, -1, :])
        generated_tokens.append(int(next_token))
        input_ids = jnp.array([[next_token]])
        position_ids = position_ids[:, -1:] + 1
        num_tokens_generated += 1
        
        # GC every 10 tokens
        if num_tokens_generated % 10 == 0:
            gc.collect()
        
        # Update peak mem
        current_mem = psutil.virtual_memory().used / (1024**3)
        if current_mem > peak_mem:
            peak_mem = current_mem
        
        if verbose:
            token = tokenizer.decode(int(next_token), skip_special_tokens=True)
            print(f"Token {num_tokens_generated}: '{token}'")
        if next_token == tokenizer.eos_token_id:
            break
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_time_per_token = total_time / num_tokens_generated if num_tokens_generated > 0 else 0
    
    print(f"\nMemory after generation: {psutil.virtual_memory().used / (1024**3):.2f} GB used")
    print(f"Peak memory during generation: {peak_mem:.2f} GB used")
    print(f"Free memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print(f"Total tokens generated: {num_tokens_generated}")
    print(f"Average time per token: {avg_time_per_token:.2f} seconds")
    
    full_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return full_output, peak_mem, avg_time_per_token

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B JAX Inference - Optimization 5: Advanced RoPE")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--verbose", action="store_true", help="Enable per-token logging")
    args = parser.parse_args()

    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    
    with open(os.path.join(args.model_path, "config.json")) as f:
        config = json.load(f)
    model = Qwen25ForCausalLM(config=config, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    params = load_params(model, args.model_path, dtype)
    
    # JIT compile model.apply
    print("Compiling JIT for speed...")
    jit_apply = jax.jit(
        model.apply, 
        static_argnames=['return_dict']
    )

    # Test prompt
    test_prompt = "Please solve this math problem step by step: If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?"
    inputs = tokenizer(test_prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    position_ids = jax.numpy.arange(input_ids.shape[1])[None, :]
    
    print(f"Input prompt: '{test_prompt}'")
    print(f"Input tokens: {input_ids[0].tolist()}")
    print(f"Input length: {len(input_ids[0])} tokens")
    
    # Generate tokens
    output, peak_mem, avg_time_per_token = generate_text(
        model, jit_apply, params, tokenizer, 15, input_ids, position_ids, args.verbose
    )
    
    print(f"\nFinal output: '{output}'")
    print(f"Peak memory: {peak_mem:.2f} GB")
    print(f"Average time per token: {avg_time_per_token:.4f} seconds")

if __name__ == "__main__":
    main() 