#!/usr/bin/env python3
"""
Qwen2.5-7B Hybrid Tensor Parallelism - 100% tensor parallelized version
Combines working ParallelDense with custom GQA-aware attention for full tensor parallelism.

Usage:
python q25j7_hybrid_tensor_parallel.py --model_path weights
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

# Set up multi-device (do this before importing jax)
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer
from flax import linen as nn
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("qwen25_hybrid_tensor_parallel")

# Global mesh for tensor parallelism
mesh = None

# --- Hybrid Tensor Parallelism Classes ---
class ParallelDense(nn.Module):
    """Parallel Dense layer for MLP and LM Head tensor parallelism."""
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
            in_specs=(
                None,
                P(None, "mp"),
            ),
            out_specs=P(None),
            check_rep=False,
        )(x, kernel)


class GQAParallelAttention(nn.Module):
    """GQA-aware parallel attention with tensor parallelism for all projections."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        batch, seq, _ = hidden_states.shape
        
        # Get attention dimensions
        hidden_size = self.config["hidden_size"]
        num_heads = self.config["num_attention_heads"]
        head_dim = hidden_size // num_heads
        num_kv_heads = self.config.get("num_key_value_heads", num_heads)
        kv_dim = num_kv_heads * head_dim
        
        # Use ParallelDense for all attention projections
        q = self.param("q_proj", ParallelDense, hidden_size, dtype=self.dtype, param_dtype=self.param_dtype, name="q_proj")(hidden_states)
        k = self.param("k_proj", ParallelDense, kv_dim, dtype=self.dtype, param_dtype=self.param_dtype, name="k_proj")(hidden_states)
        v = self.param("v_proj", ParallelDense, kv_dim, dtype=self.dtype, param_dtype=self.param_dtype, name="v_proj")(hidden_states)
        
        # Reshape to attention format
        q = q.reshape(batch, seq, num_heads, head_dim)
        k = k.reshape(batch, seq, num_kv_heads, head_dim)
        v = v.reshape(batch, seq, num_kv_heads, head_dim)

        # Apply rotary embeddings
        if position_ids is not None:
            cos, sin = compute_cos_sin_cache(position_ids, head_dim, self.config.get("rope_theta", 1000000.0))
            q, k = apply_rotary_emb(q, k, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = jnp.concatenate([past_k, k], axis=1)
            v = jnp.concatenate([past_v, v], axis=1)

        cache_k, cache_v = k, v

        # GQA: Repeat k/v to match query heads
        if num_heads != num_kv_heads:
            repeat = num_heads // num_kv_heads
            k = jnp.repeat(k, repeat, axis=2)
            v = jnp.repeat(v, repeat, axis=2)

        # Attention computation
        q = q.transpose(0, 2, 1, 3)  # [batch, heads, seq, head_dim]
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = 1.0 / jnp.sqrt(head_dim)
        scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
        if attention_mask is not None:
            scores += attention_mask
        # Use higher precision for attention scores to reduce FP diffs
        scores = scores.astype(jnp.float64)
        probs = jnp.clip(jax.nn.softmax(scores.astype(jnp.float32), axis=-1), 1e-9, 1 - 1e-9)
        attn_out = jnp.einsum('bhqk,bhkd->bhqd', probs, v)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq, hidden_size)

        # Use ParallelDense for output projection
        return self.param("o_proj", ParallelDense, hidden_size, dtype=self.dtype, param_dtype=self.param_dtype, use_bias=False, name="o_proj")(attn_out), (cache_k, cache_v)


class QwenMLP(nn.Module):
    """MLP with tensor parallelism using ParallelDense."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        c = self.config
        intermediate_size = c.get("intermediate_size", 4 * c["hidden_size"])
        
        # Use ParallelDense for all MLP layers
        gate_proj = self.param("gate_proj", ParallelDense, intermediate_size, dtype=self.dtype, param_dtype=self.param_dtype, name="gate_proj")(x)
        up_proj = self.param("up_proj", ParallelDense, intermediate_size, dtype=self.dtype, param_dtype=self.param_dtype, name="up_proj")(x)
        down_proj = self.param("down_proj", ParallelDense, c["hidden_size"], dtype=self.dtype, param_dtype=self.param_dtype, name="down_proj")(jax.nn.silu(gate_proj) * up_proj)
        
        return down_proj


class QwenDecoderLayer(nn.Module):
    """Decoder layer with hybrid tensor parallelism."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        residual = hidden_states
        hidden_states = self.param("input_layernorm", nn.RMSNorm, epsilon=self.config.get("rms_norm_eps", 1e-6), dtype=self.dtype, name="input_layernorm")(hidden_states)
        hidden_states, past_key_value = self.param("self_attn", GQAParallelAttention, self.config, dtype=self.dtype, param_dtype=self.param_dtype)(hidden_states, attention_mask, position_ids, past_key_value)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.param("post_attention_layernorm", nn.RMSNorm, epsilon=self.config.get("rms_norm_eps", 1e-6), dtype=self.dtype, name="post_attention_layernorm")(hidden_states)
        hidden_states = residual + self.param("mlp", QwenMLP, self.config, dtype=self.dtype, param_dtype=self.param_dtype)(hidden_states)
        return hidden_states, past_key_value


class Qwen25ForCausalLM(nn.Module):
    """Qwen2.5 model with 100% hybrid tensor parallelism."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, return_dict=True):
        batch, seq = input_ids.shape
        key_len = seq if past_key_values is None or past_key_values[0] is None else past_key_values[0][0].shape[1] + seq

        if attention_mask is None:
            attention_mask = jnp.ones((batch, 1, seq, key_len), dtype=self.dtype)
        causal_mask = make_causal_mask(seq, key_len)[None, None, :, :]
        attention_bias = jnp.where(attention_mask == 0, -1e9, 0) + causal_mask

        hidden_states = self.param("embed_tokens", nn.Embed, self.config["vocab_size"], self.config["hidden_size"], dtype=self.dtype)(input_ids)
        if past_key_values is None:
            past_key_values = [None] * self.config["num_hidden_layers"]

        new_key_values = []
        
        for i in range(self.config["num_hidden_layers"]):
            layer = self.param(f"layers_{i}", QwenDecoderLayer, self.config, dtype=self.dtype, param_dtype=self.param_dtype)
            hidden_states, new_kv = layer(hidden_states, attention_bias, position_ids, past_key_values[i])
            new_key_values.append(new_kv)

        hidden_states = self.param("norm", nn.RMSNorm, epsilon=self.config.get("rms_norm_eps", 1e-6), dtype=self.dtype)(hidden_states)
        
        # Use ParallelDense for lm_head
        logits = self.param("lm_head", ParallelDense, self.config["vocab_size"], dtype=self.dtype, param_dtype=self.param_dtype)(hidden_states)

        if return_dict:
            return {"logits": logits, "past_key_values": new_key_values}
        return logits


# --- Utility Functions ---
def compute_cos_sin_cache(position_ids, head_dim, rope_theta=1000000.0):
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
    # q, k: [batch, seq, heads, head_dim]
    # cos, sin: [batch, seq, 1, dim] where dim = head_dim // 2
    half_dim = q.shape[-1] // 2
    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    k1, k2 = k[..., :half_dim], k[..., half_dim:]
    # cos and sin are already [batch, seq, 1, dim], so they broadcast correctly
    q_rot = jnp.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
    k_rot = jnp.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)
    return q_rot, k_rot

def make_causal_mask(q_len, k_len):
    i = jnp.arange(q_len)[:, None]
    j = jnp.arange(k_len)[None, :]
    return jnp.where(i >= j - (k_len - q_len), 0, -1e9)

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

def load_params_hybrid(model, model_path, dtype):
    """Load model parameters from safetensors files into hybrid model."""
    print(f"[Progress] Loading hybrid model weights from {model_path}...")
    loaded_count = 0
    
    params = {}
    
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            with safe_open(os.path.join(model_path, file), framework="numpy") as f:
                for key in f.keys():
                    path = get_param_path(key)
                    if path:
                        param = f.get_tensor(key)
                        param = jnp.array(param, dtype=jnp.bfloat16)
                        param = transpose_if_needed(key, param)
                        
                        # Build nested dictionary structure
                        d = params
                        for p in path[:-1]:
                            if p not in d:
                                d[p] = {}
                            d = d[p]
                        d[path[-1]] = param
                        
                        print(f"[Progress] Loaded: {key} -> {path}")
                        loaded_count += 1
    
    print(f"[Progress] Weight loading completed. Loaded {loaded_count} parameters.")
    return params

def sample_next_token(logits):
    return jnp.argmax(logits, axis=-1)

def generate_text_hybrid(model, params, tokenizer, max_tokens, prompt):
    print("[Progress] Starting hybrid text generation...")
    
    # Monitor memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024**3
    print(f"[Progress] Initial memory before generation: {initial_memory:.2f} GB used")
    print(f"[Progress] Free memory: {psutil.virtual_memory().available / 1024**3:.2f} GB")
    
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="jax")
    batch, seq = input_ids.shape
    position_ids = jnp.arange(seq, dtype=jnp.int32)[None, :]
    past_key_values = None
    
    generated_tokens = []
    start_time = time.time()
    peak_memory = initial_memory
    
    num_tokens_generated = 0
    print(f"[Progress] Entering generation loop for {max_tokens} tokens...")
    
    for i in range(max_tokens):
        print(f"[Progress] Generating token {i+1}/{max_tokens}...")
        
        # Create attention mask with proper shape for current sequence
        current_seq_len = input_ids.shape[1]
        key_len = current_seq_len if past_key_values is None or past_key_values[0] is None else past_key_values[0][0].shape[1] + current_seq_len
        attention_mask = jnp.ones((batch, 1, current_seq_len, key_len), dtype=jnp.float32)
        
        # Use model.apply directly since all layers handle tensor parallelism internally
        outputs = model.apply(params, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values, return_dict=True)
        logits = outputs["logits"]
        past_key_values = outputs["past_key_values"]
        
        next_token = sample_next_token(logits[:, -1, :])
        generated_tokens.append(int(next_token))
        input_ids = jnp.array([[next_token]])
        position_ids = position_ids[:, -1:] + 1
        num_tokens_generated += 1
        
        # Update peak mem
        current_mem = psutil.virtual_memory().used / (1024**3)
        if current_mem > peak_memory:
            peak_memory = current_mem
        
        token = tokenizer.decode(int(next_token), skip_special_tokens=True)
        print(f"[Progress] Gen token {i+1}/{max_tokens}: {token}")
        
        # Check for EOS
        if int(next_token) == tokenizer.eos_token_id:
            print(f"[Progress] EOS token generated, stopping.")
            break
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_token = total_time / num_tokens_generated if num_tokens_generated > 0 else 0
    
    print(f"[Progress] Memory after generation: {psutil.virtual_memory().used / (1024**3):.2f} GB used")
    print(f"[Progress] Peak memory during generation: {peak_memory:.2f} GB used")
    print(f"[Progress] Free memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print(f"[Progress] Total tokens generated: {num_tokens_generated}")
    print(f"[Progress] Average time per token: {avg_time_per_token:.4f} seconds")
    
    full_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print("[Progress] Generation complete.")
    return full_output, peak_memory, avg_time_per_token

def extract_answer(output: str) -> str:
    """Extract the answer from the model output."""
    # Simple extraction - look for the last meaningful response
    lines = output.strip().split('\n')
    for line in reversed(lines):
        if line.strip() and not line.startswith('system') and not line.startswith('user'):
            return line.strip()
    return output.strip()

def setup_device_mesh():
    """Setup device mesh for tensor parallelism."""
    global mesh
    print("[Progress] Setting up device mesh...")
    devices = jax.devices()
    print(f"[Progress] Available devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device}")
    mesh = Mesh(devices, axis_names=("mp",))
    print(f"[Progress] Created mesh: {mesh}")
    return mesh

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B Hybrid Tensor Parallel Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    args = parser.parse_args()
    
    # Setup device mesh
    mesh = setup_device_mesh()
    
    # Load config
    config_path = os.path.join(args.model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Create hybrid model with 100% tensor parallelism
    print("[Progress] Creating hybrid model with 100% tensor parallelism...")
    model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
    
    # Load parameters
    params = load_params_hybrid(model, args.model_path, jnp.bfloat16)
    
    print("\n=== HYBRID TENSOR PARALLELISM TEST ===")
    test_sample = {"question": "What is 2 + 2?", "answer": "4"}
    print(f"Testing with: {test_sample['question']}")
    
    # Generate 5 tokens for testing hybrid tensor parallelism benefits
    output, peak_mem, avg_time_per_token = generate_text_hybrid(model, params, tokenizer, 5, test_sample["question"])
    print(f"Output: {output}")
    print(f"Peak memory: {peak_mem:.2f} GB")
    print(f"Avg time per token: {avg_time_per_token:.4f} seconds")
    print("=== HYBRID TEST COMPLETE ===")

if __name__ == "__main__":
    main() 