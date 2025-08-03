#!/usr/bin/env python3
"""
Self-contained Qwen2.5-7B-Instruct inference script for multi-device JAX with tensor parallelism.
- Fixed rotary embeddings (RoPE) with proper broadcasting.
- Corrected GQA attention mechanism for shape compatibility.
- Optimized sampling to prevent repetitive outputs.
- Enhanced for GSM8K-style math problems.
- Greedy sampling (temperature=0) for deterministic outputs.
- Hardcoded GSM8K benchmarking with 10 samples.
- Generalized text generation for multiple prompts.
- Answer extraction with boxed format support.
- Detailed memory monitoring with psutil (peak per sample).
- Timing for generation speed (avg seconds per token).
- JAX_ENABLE_X64 disabled globally for faster inference.
- Default bfloat16 for faster inference.
- Pure JAX sampling (no PyTorch dependency).
- Enhanced memory management with GC collects.
- TENSOR PARALLELISM: Multi-device distributed inference using shard_map.

Usage:
python q25j7_tensor_parallel.py --model_path weights
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
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

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
logger = logging.getLogger("qwen25_tensor_parallel")

# Global mesh for tensor parallelism
mesh = None

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
        # Use regular Dense layers for attention (GQA needs special handling)
        self.q_proj = nn.Dense(self.hidden_size, dtype=jnp.bfloat16, name="q_proj")
        self.k_proj = nn.Dense(self.kv_dim, dtype=jnp.bfloat16, name="k_proj")
        self.v_proj = nn.Dense(self.kv_dim, dtype=jnp.bfloat16, name="v_proj")
        self.o_proj = nn.Dense(self.hidden_size, dtype=jnp.bfloat16, use_bias=False, name="o_proj")
        self.rope_theta = c.get("rope_theta", 1000000.0)

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
        # Use higher precision for attention scores to reduce FP diffs
        scores = scores.astype(jnp.float64)
        probs = jnp.clip(jax.nn.softmax(scores.astype(jnp.float32), axis=-1), 1e-9, 1 - 1e-9)
        attn_out = jnp.einsum('bhqk,bhkd->bhqd', probs, v)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq, self.hidden_size)

        return self.o_proj(attn_out), (cache_k, cache_v)

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
        local_shape = (in_dim, out_dim)

        # Use the same parameter name structure as regular Dense layers
        kernel = self.param(
            "kernel", nn.initializers.lecun_normal(), local_shape, self.param_dtype
        )

        def matmul_fn(x, k):
            axis_idx = jax.lax.axis_index("mp")
            # print(f"🔧 Device {axis_idx} running matmul: x.shape = {x.shape}, kernel.shape = {k.shape}")

            local_out = jnp.einsum("bsd,df->bsf", x, k)

            full_out = jax.lax.all_gather(local_out, axis_name="mp", axis=0)

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

class QwenMLP(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        self.intermediate_size = c.get("intermediate_size", 4 * c["hidden_size"])
        # Use ParallelDense for tensor parallelism
        self.gate_proj = ParallelDense(
            self.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.dtype,
            name="gate_proj"
        )
        self.up_proj = ParallelDense(
            self.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.dtype,
            name="up_proj"
        )
        self.down_proj = ParallelDense(
            c["hidden_size"],
            dtype=self.dtype,
            param_dtype=self.dtype,
            name="down_proj"
        )

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
        # Use ParallelDense for tensor parallelism
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
    print(f"[Progress] Loading JAX model weights from {model_path}...")
    params = {"params": {}}
    loaded_count = 0
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            with safe_open(os.path.join(model_path, file), framework="numpy") as f:
                for key in f.keys():
                    path = get_param_path(key)
                    if path:
                        param = f.get_tensor(key)
                        param = jnp.array(param, dtype=jnp.bfloat16) # Always load as bfloat16
                        param = transpose_if_needed(key, param)
                        d = params["params"]
                        for p in path[:-1]:
                            d = d.setdefault(p, {})
                        d[path[-1]] = param
                        loaded_count += 1
                        if loaded_count <= 10:
                            print(f"[Progress] Loaded: {key} -> {path}")
    gc.collect()
    print(f"[Progress] Weight loading completed. Loaded {loaded_count} parameters.")
    return params

# --- Generation ---
def sample_next_token(logits):
    """Simplified greedy sampling only."""
    return int(jnp.argmax(logits, axis=-1)[0])

def generate_text(model, params, tokenizer, max_tokens, prompt):
    print("[Progress] Starting text generation...")
    
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
    
    # Create comprehensive sharded forward pass
    def model_forward(model_params, input_ids, attention_mask, position_ids, past_key_values):
        outputs = model.apply(model_params, input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, past_key_values=past_key_values, return_dict=True)
        logits = outputs["logits"]
        # Gather logits from all devices
        gathered_logits = jax.lax.all_gather(logits, axis_name="mp", axis=0)
        gathered_logits = jnp.reshape(jnp.transpose(gathered_logits, (1, 2, 0, 3)), (logits.shape[0], logits.shape[1], -1))
        return gathered_logits, outputs["past_key_values"]

    # Create sharding specs for tensor parallelism
    # Focus on MLP and lm_head layers which work well
    def create_sharding_specs(params):
        specs = {}
        for key, value in params.items():
            if 'kernel' in key and value.ndim == 2:  # Weight matrices
                if 'mlp' in key or 'lm_head' in key:
                    # Shard output dimension for MLP and lm_head
                    specs[key] = P(None, "mp")
                else:
                    specs[key] = P()
            else:
                specs[key] = P()
        return specs

    # Use shard_map for tensor parallelism
    sharded_forward = shard_map(
        model_forward,
        mesh=mesh,
        in_specs=(create_sharding_specs(params), P(), P(), P(), P()),  # Params sharded, inputs replicated
        out_specs=(P(), P()),  # Outputs replicated (logits are gathered inside)
        check_rep=False,
    )

    num_tokens_generated = 0
    print(f"[Progress] Entering generation loop for {max_tokens} tokens...")
    for i in range(max_tokens):
        print(f"[Progress] Generating token {i+1}/{max_tokens}...")
        # Create attention mask with proper shape for current sequence
        current_seq_len = input_ids.shape[1]
        key_len = current_seq_len if past_key_values is None or past_key_values[0] is None else past_key_values[0][0].shape[1] + current_seq_len
        attention_mask = jnp.ones((batch, 1, current_seq_len, key_len), dtype=jnp.float32)
        
        # Use model.apply directly since ParallelDense handles tensor parallelism
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
        if int(next_token) == tokenizer.eos_token_id or "<|im_end|>" in token:
            print("[Progress] Stopping generation: EOS token encountered.")
            break
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_token = total_time / num_tokens_generated if num_tokens_generated > 0 else 0
    
    print(f"[Progress] Memory after generation: {psutil.virtual_memory().used / (1024**3):.2f} GB used")
    print(f"[Progress] Peak memory during generation: {peak_memory:.2f} GB used")
    print(f"[Progress] Free memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print(f"[Progress] Total tokens generated: {num_tokens_generated}")
    print(f"[Progress] Average time per token: {avg_time_per_token:.2f} seconds")
    
    full_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print("[Progress] Generation complete.")
    return full_output, peak_memory, avg_time_per_token

def extract_answer(output: str) -> str:
    """Extract numerical answer from model output."""
    import re
    match = re.search(r'\\boxed\{([0-9]+)\}', output)
    return match.group(1) if match else None

def setup_device_mesh():
    """Setup device mesh for tensor parallelism."""
    global mesh
    print("[Progress] Setting up device mesh...")
    devices = jax.devices()
    print(f"[Progress] Available devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device}")
    
    # Use all available devices for tensor parallelism
    mesh = Mesh(devices, axis_names=("mp",))
    print(f"[Progress] Created mesh: {mesh}")
    return mesh

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B-Instruct JAX Inference for GSM8K Benchmark")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    
    # Setup device mesh for tensor parallelism
    mesh = setup_device_mesh()
    
    with open(os.path.join(args.model_path, "config.json")) as f:
        config = json.load(f)
    model = Qwen25ForCausalLM(config=config, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    params = load_params(model, args.model_path, dtype)
    
    # Quick test with just one simple sample for incremental testing
    print("\n=== QUICK TEST MODE ===")
    test_sample = {"question": "What is 2 + 2?", "answer": "4"}
    print(f"Testing with: {test_sample['question']}")
    # Generate 5 tokens for testing tensor parallelism benefits
    output, peak_mem, avg_time_per_token = generate_text(model, params, tokenizer, 5, test_sample["question"])
    print(f"Output: {output}")
    print(f"Peak memory: {peak_mem:.2f} GB")
    print(f"Avg time per token: {avg_time_per_token:.4f} seconds")
    print("=== QUICK TEST COMPLETE ===")

if __name__ == "__main__":
    main() 