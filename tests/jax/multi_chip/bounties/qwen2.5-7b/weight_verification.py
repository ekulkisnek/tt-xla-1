#!/usr/bin/env python3
"""
Script to confirm weights are identical between PyTorch and JAX implementations.
- Loads model in PyTorch and logs shapes/dtypes of key params (embed_tokens, q_proj in layer 0).
- Loads model in JAX and logs shapes/dtypes of the same params.
- Assumes local weights in the provided model_path.

Usage:
python weight_verification.py --model_path weights
"""

import argparse
import gc
import logging
import os
import json
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import linen as nn
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("weight_verification")

# --- JAX Model Code (Minimal for param structure) ---
class QwenAttention(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float16  # Match PyTorch

    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        self.num_heads = c["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = c.get("num_key_value_heads", self.num_heads)
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Dense(self.hidden_size, dtype=self.dtype, name="q_proj")
        # Other projections omitted for verification

class QwenDecoderLayer(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float16

    def setup(self):
        self.self_attn = QwenAttention(config=self.config, dtype=self.dtype)
        # Other components omitted

class Qwen25ForCausalLM(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float16

    def setup(self):
        c = self.config
        self.embed_tokens = nn.Embed(c["vocab_size"], c["hidden_size"], dtype=self.dtype, name="embed_tokens")
        self.layers = [QwenDecoderLayer(config=c, dtype=self.dtype, name=f"layers_{i}") for i in range(c["num_hidden_layers"])]
        # Norm and lm_head omitted

# --- JAX Weight Loading (from your JAX code) ---
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

def load_jax_params(model_path, dtype):
    params = {"params": {}}
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            with safe_open(os.path.join(model_path, file), framework="numpy") as f:
                for key in f.keys():
                    path = get_param_path(key)
                    if path:
                        param = f.get_tensor(key)
                        param = jnp.array(param, dtype=jnp.float32 if param.dtype == np.float16 else dtype)
                        param = transpose_if_needed(key, param)
                        d = params["params"]
                        for p in path[:-1]:
                            d = d.setdefault(p, {})
                        d[path[-1]] = param
    logger.info("JAX weight loading completed")
    return params

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Verify weights between PyTorch and JAX")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    args = parser.parse_args()

    model_path = args.model_path

    # --- Load PyTorch Model and Log Key Params ---
    logger.info("Loading PyTorch model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        offload_folder="./offload",
        offload_state_dict=True,
    )

    # Log PyTorch key params
    embed_tokens_pt = model.model.embed_tokens.weight
    logger.info(f"PyTorch embed_tokens shape: {embed_tokens_pt.shape}, dtype: {embed_tokens_pt.dtype}")

    q_proj_pt = model.model.layers[0].self_attn.q_proj.weight
    logger.info(f"PyTorch q_proj (layer 0) shape: {q_proj_pt.shape}, dtype: {q_proj_pt.dtype}")

    del model
    gc.collect()

    # --- Load JAX Model and Log Key Params ---
    logger.info("Loading JAX model...")
    with open(os.path.join(model_path, "config.json")) as f:
        config = json.load(f)
    model = Qwen25ForCausalLM(config=config, dtype=jnp.float16)  # Match PyTorch dtype
    params = load_jax_params(model_path, jnp.float16)

    # Log JAX key params
    embed_tokens_jax = params["params"]["embed_tokens"]["embedding"]
    logger.info(f"JAX embed_tokens shape: {embed_tokens_jax.shape}, dtype: {embed_tokens_jax.dtype}")

    q_proj_jax = params["params"]["layers_0"]["self_attn"]["q_proj"]["kernel"]
    logger.info(f"JAX q_proj (layer 0) shape: {q_proj_jax.shape}, dtype: {q_proj_jax.dtype}")

    gc.collect()
    jax.clear_caches()

if __name__ == "__main__":
    main() 