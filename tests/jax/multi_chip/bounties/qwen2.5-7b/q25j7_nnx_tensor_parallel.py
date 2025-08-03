#!/usr/bin/env python3
"""
Qwen2.5-7B with NNX Tensor Parallelism - 100% tensor parallelized version
Based on Mixtral's successful NNX implementation for comprehensive tensor parallelism.

Usage:
python q25j7_nnx_tensor_parallel.py --model_path weights
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
from flax import nnx
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("qwen25_nnx_tensor_parallel")

# Global mesh for tensor parallelism
mesh = None

# --- NNX Model Code ---
class QwenNNXAttention(nnx.Module):
    """NNX-based attention with automatic tensor parallelism."""
    
    def __init__(self, config: Dict[str, Any], dtype: jnp.dtype, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.dtype = dtype
        
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.get("num_key_value_heads", self.num_heads)
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.rope_theta = config.get("rope_theta", 1000000.0)
        
        # Use NNX Linear with automatic sharding for ALL attention layers
        self.q_proj = nnx.Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, "X")),
            rngs=rngs,
        )
        
        self.k_proj = nnx.Linear(
            in_features=self.hidden_size,
            out_features=self.kv_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, "X")),
            rngs=rngs,
        )
        
        self.v_proj = nnx.Linear(
            in_features=self.hidden_size,
            out_features=self.kv_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, "X")),
            rngs=rngs,
        )
        
        self.o_proj = nnx.Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("X", None)),
            rngs=rngs,
        )

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        batch, seq, _ = hidden_states.shape

        # Project inputs with NNX Linear layers
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


class QwenNNXMLP(nnx.Module):
    """NNX-based MLP with automatic tensor parallelism."""
    
    def __init__(self, config: Dict[str, Any], dtype: jnp.dtype, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.dtype = dtype
        
        self.intermediate_size = config.get("intermediate_size", 4 * config["hidden_size"])
        
        # Use NNX Linear with automatic sharding for ALL MLP layers
        self.gate_proj = nnx.Linear(
            in_features=config["hidden_size"],
            out_features=self.intermediate_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, "X")),
            rngs=rngs,
        )
        
        self.up_proj = nnx.Linear(
            in_features=config["hidden_size"],
            out_features=self.intermediate_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, "X")),
            rngs=rngs,
        )
        
        self.down_proj = nnx.Linear(
            in_features=self.intermediate_size,
            out_features=config["hidden_size"],
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("X", None)),
            rngs=rngs,
        )

    def __call__(self, x):
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenNNXDecoderLayer(nnx.Module):
    """NNX-based decoder layer with automatic tensor parallelism."""
    
    def __init__(self, config: Dict[str, Any], dtype: jnp.dtype, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.dtype = dtype
        
        # Use NNX modules for all components
        self.input_layernorm = nnx.RMSNorm(
            dim=config["hidden_size"],
            eps=config.get("rms_norm_eps", 1e-6),
            dtype=self.dtype,
        )
        self.self_attn = QwenNNXAttention(config=config, dtype=self.dtype, rngs=rngs)
        self.post_attention_layernorm = nnx.RMSNorm(
            dim=config["hidden_size"],
            eps=config.get("rms_norm_eps", 1e-6),
            dtype=self.dtype,
        )
        self.mlp = QwenNNXMLP(config=config, dtype=self.dtype, rngs=rngs)

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, past_key_value = self.self_attn(hidden_states, attention_mask, position_ids, past_key_value)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, past_key_value


class Qwen25NNXForCausalLM(nnx.Module):
    """NNX-based Qwen2.5 model with 100% tensor parallelism."""
    
    def __init__(self, config: Dict[str, Any], dtype: jnp.dtype, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.dtype = dtype
        
        # Use NNX modules for all components
        self.embed_tokens = nnx.Embed(
            num_embeddings=config["vocab_size"],
            features=config["hidden_size"],
            dtype=self.dtype,
            embedding_init=nnx.with_partitioning(nnx.initializers.normal(0.02), ("X", None)),
            rngs=rngs,
        )
        
        self.layers = [
            QwenNNXDecoderLayer(config=config, dtype=self.dtype, rngs=rngs)
            for _ in range(config["num_hidden_layers"])
        ]
        
        self.norm = nnx.RMSNorm(
            dim=config["hidden_size"],
            eps=config.get("rms_norm_eps", 1e-6),
            dtype=self.dtype,
        )
        
        # Use NNX Linear with automatic sharding for lm_head
        self.lm_head = nnx.Linear(
            in_features=config["hidden_size"],
            out_features=config["vocab_size"],
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, "X")),
            rngs=rngs,
        )

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

    def prepare_sharding(self):
        """Prepare sharding specifications for comprehensive tensor parallelism."""
        state = nnx.state(self)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(self, sharded_state)
        
        # KV cache specifications
        cache_specs = {}
        for i in range(self.config["num_hidden_layers"]):
            layer_key = f"layer_{i}"
            cache_specs[layer_key] = {
                "cached_key": P(),  # Replicated
                "cached_value": P(),  # Replicated
                "cache_index": P(),  # Replicated
            }
        
        return cache_specs, pspecs


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

def load_params_nnx(model, model_path, dtype):
    """Load model parameters from safetensors files into NNX model."""
    print(f"[Progress] Loading NNX model weights from {model_path}...")
    loaded_count = 0
    
    # Get the current state
    state = nnx.state(model)
    
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            with safe_open(os.path.join(model_path, file), framework="numpy") as f:
                for key in f.keys():
                    path = get_param_path(key)
                    if path:
                        param = f.get_tensor(key)
                        param = jnp.array(param, dtype=jnp.bfloat16)
                        param = transpose_if_needed(key, param)
                        
                        # Update the state with the loaded parameter
                        d = state
                        for p in path[:-1]:
                            d = d[p]
                        d[path[-1]] = param
                        
                        print(f"[Progress] Loaded: {key} -> {path}")
                        loaded_count += 1
    
    # Update the model with the loaded state
    nnx.update(model, state)
    print(f"[Progress] Weight loading completed. Loaded {loaded_count} parameters.")
    return model

def sample_next_token(logits):
    return jnp.argmax(logits, axis=-1)

def generate_text_nnx(model, tokenizer, max_tokens, prompt):
    print("[Progress] Starting NNX text generation...")
    
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
    
    # Prepare sharding for comprehensive tensor parallelism
    cache_specs, model_specs = model.prepare_sharding()
    
    # Create sharded forward pass (Mixtral approach)
    def model_forward(model_state, input_ids, attention_mask, position_ids, past_key_values):
        # Merge the model with the state
        model_merged = nnx.merge(nnx.graphdef(model), model_state)
        
        # Call the model
        outputs = model_merged(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            return_dict=True
        )
        return outputs["logits"], outputs["past_key_values"]

    # Use nnx.shard_map for comprehensive tensor parallelism
    with mesh:
        sharded_forward = nnx.shard_map(
            model_forward,
            mesh=mesh,
            in_specs=(model_specs, P(), P(), P(), cache_specs),
            out_specs=(P(None, None, "X"), cache_specs),
            check_rep=False,
        )
        
        jit_forward = nnx.jit(sharded_forward, donate_argnums=(4,))
        
        # Get initial model state
        model_state = nnx.state(model)
        
        num_tokens_generated = 0
        print(f"[Progress] Entering generation loop for {max_tokens} tokens...")
        
        for i in range(max_tokens):
            print(f"[Progress] Generating token {i+1}/{max_tokens}...")
            
            # Create attention mask with proper shape for current sequence
            current_seq_len = input_ids.shape[1]
            key_len = current_seq_len if past_key_values is None or past_key_values[0] is None else past_key_values[0][0].shape[1] + current_seq_len
            attention_mask = jnp.ones((batch, 1, current_seq_len, key_len), dtype=jnp.float32)
            
            # Use sharded forward pass for comprehensive tensor parallelism
            logits, past_key_values = jit_forward(
                model_state,
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
            )
            
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
    mesh = Mesh(devices, axis_names=("X",))
    print(f"[Progress] Created mesh: {mesh}")
    return mesh

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B NNX Tensor Parallel Inference")
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
    
    # Create NNX model with comprehensive tensor parallelism
    print("[Progress] Creating NNX model with 100% tensor parallelism...")
    rngs = nnx.Rngs(0)
    model = Qwen25NNXForCausalLM(config=config, dtype=jnp.bfloat16, rngs=rngs)
    
    # Load parameters
    model = load_params_nnx(model, args.model_path, jnp.bfloat16)
    
    print("\n=== NNX TENSOR PARALLELISM TEST ===")
    test_sample = {"question": "What is 2 + 2?", "answer": "4"}
    print(f"Testing with: {test_sample['question']}")
    
    # Generate 5 tokens for testing NNX tensor parallelism benefits
    output, peak_mem, avg_time_per_token = generate_text_nnx(model, tokenizer, 5, test_sample["question"])
    print(f"Output: {output}")
    print(f"Peak memory: {peak_mem:.2f} GB")
    print(f"Avg time per token: {avg_time_per_token:.4f} seconds")
    print("=== NNX TEST COMPLETE ===")

if __name__ == "__main__":
    main() 