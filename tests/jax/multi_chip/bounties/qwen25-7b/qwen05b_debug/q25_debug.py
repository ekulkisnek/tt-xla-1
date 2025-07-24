#!/usr/bin/env python3
"""
Self-contained, real Qwen2.5-7B-Instruct inference script for single-device JAX.
- Uses real model code and weight mapping from run_inference.py/model.py
- Prints output to terminal only
- Allows dtype selection (bfloat16/float32)
- Cleans up memory after each run
- No file output, no simplification, no external local imports
- Uses Qwen2.5-7B-INSTRUCT weights and applies chat templates

Usage 
python q25_jax_instruct.py --model_path ../instruct_weights --prompt "Hello, how are you?" --max_tokens 20 --temperature 0.7 --top_p 0.8 --top_k 20 --dtype bfloat16
python q25_jax_instruct.py --model_path ../instruct_weights --prompt "The capital of France is" --max_tokens 10 --temperature 0.1
"""
import os
import sys
import time
import json
import gc
import argparse
import logging
from typing import Dict, Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from safetensors import safe_open

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("qwen25_instruct")

# --- Model code (copied from your real model.py, single-device only) ---
class QwenAttention(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        self.num_heads = c["num_attention_heads"]
        self.head_dim = c.get("head_dim", self.hidden_size // self.num_heads)
        self.num_kv_heads = c.get("num_key_value_heads", self.num_heads)
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Dense(self.hidden_size, dtype=self.dtype, name="q_proj")
        self.k_proj = nn.Dense(self.kv_dim, dtype=self.dtype, name="k_proj")
        self.v_proj = nn.Dense(self.kv_dim, dtype=self.dtype, name="v_proj")
        self.o_proj = nn.Dense(self.hidden_size, dtype=self.dtype, use_bias=False, name="o_proj")
        self.rope_theta = c.get("rope_theta", 10000.0)
        self.max_position_embeddings = c.get("max_position_embeddings", 4096)
    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, cos=None, sin=None):
        batch, seq, _ = hidden_states.shape
        
        # Project current hidden states
        q = self.q_proj(hidden_states).reshape(batch, seq, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(batch, seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(batch, seq, self.num_kv_heads, self.head_dim)
        
        # Apply rotary embeddings
        if position_ids is not None:
            if cos is None or sin is None:
                cos, sin = compute_cos_sin_cache(position_ids, self.head_dim, self.rope_theta)
            q, k = apply_rotary_emb(q, k, cos, sin)
        
        # Handle past key-value cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            
            # Ensure past cache is in the same format as current k,v: [batch, seq, num_kv_heads, head_dim]
            if past_k.shape[1] == self.num_kv_heads and past_k.shape[2] != self.num_kv_heads:
                # Past cache is in transposed format [batch, num_kv_heads, seq, head_dim], transpose it back
                past_k = jnp.transpose(past_k, (0,2,1,3))
                past_v = jnp.transpose(past_v, (0,2,1,3))
            
            # Convert past cache to KV head format if needed
            if past_k.shape[2] == self.num_heads:  # If past cache is in query head format
                # Reshape to group query heads into KV heads
                past_k = past_k.reshape(batch, -1, self.num_kv_heads, self.num_heads // self.num_kv_heads, self.head_dim)
                past_k = jnp.mean(past_k, axis=3)  # Average over query heads per KV head
                past_v = past_v.reshape(batch, -1, self.num_kv_heads, self.num_heads // self.num_kv_heads, self.head_dim)
                past_v = jnp.mean(past_v, axis=3)
            elif past_k.shape[0] == 0 or past_k.shape[1] == 0:  # Empty cache
                past_k = jnp.zeros((batch, 0, self.num_kv_heads, self.head_dim), dtype=past_k.dtype)
                past_v = jnp.zeros((batch, 0, self.num_kv_heads, self.head_dim), dtype=past_v.dtype)
            elif past_k.shape[2] != self.num_kv_heads:  # Unexpected head count
                raise ValueError(f"Past cache has unexpected number of heads: {past_k.shape[2]}, expected {self.num_kv_heads}")
            
            # Concatenate along sequence dimension
            k = jnp.concatenate([past_k, k], axis=1)
            v = jnp.concatenate([past_v, v], axis=1)
        
        # Store cache before repeating (should be in KV head format)
        cache_k = k
        cache_v = v
        
        # GQA: repeat k/v to match query heads for attention computation
        if self.num_heads != self.num_kv_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = jnp.repeat(k, repeat, axis=2)
            v = jnp.repeat(v, repeat, axis=2)
        
        # Transpose for attention: [b, h, s, d]
        q = jnp.transpose(q, (0,2,1,3))  # [batch, num_heads, seq, head_dim]
        k = jnp.transpose(k, (0,2,1,3))  # [batch, num_heads, seq, head_dim]
        v = jnp.transpose(v, (0,2,1,3))  # [batch, num_heads, seq, head_dim]
        
        # Attention
        scale = 1.0 / np.sqrt(self.head_dim)
        attn_scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Force FP32 softmax for numerical stability (checkpoint 9)
        attn_scores = attn_scores.astype(jnp.float32)
        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        attn_out = jnp.einsum('bhqk,bhkd->bhqd', attn_probs, v)
        attn_out = jnp.transpose(attn_out, (0,2,1,3)).reshape(batch, seq, self.hidden_size)
        
        # Return both output and updated cache (in same format as input: [batch, seq, num_kv_heads, head_dim])
        return self.o_proj(attn_out), (cache_k, cache_v)

def compute_cos_sin_cache(position_ids, head_dim, rope_theta=10000.0):
    # position_ids: [batch, seq]
    # Returns cos, sin: [batch, seq, head_dim]
    pos = np.array(position_ids)
    if pos.ndim == 1:
        pos = pos[None, :]
    dim = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, dim, dtype=np.float32) / dim))
    freqs = np.einsum('bi,j->bij', pos, inv_freq)
    
    # Create cos and sin for full head_dim (not head_dim//2)
    cos = jnp.array(np.cos(freqs))
    sin = jnp.array(np.sin(freqs))
    
    # Repeat to match head_dim
    cos = jnp.repeat(cos, 2, axis=-1)
    sin = jnp.repeat(sin, 2, axis=-1)
    
    return cos, sin

def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)

def apply_rotary_emb(q, k, cos, sin):
    # q, k: [batch, seq, n_heads, head_dim]
    # cos, sin: [batch, seq, head_dim]
    def _rope(x, cos, sin):
        # Reshape cos/sin to match x's dimensions
        cos = cos[..., None, :]  # [batch, seq, 1, head_dim]
        sin = sin[..., None, :]  # [batch, seq, 1, head_dim]
        return (x * cos) + (rotate_half(x) * sin)
    return _rope(q, cos, sin), _rope(k, cos, sin)

def make_causal_mask(q_len, k_len, *, dtype):
    """Create proper causal mask - always use FP32 for numerical stability."""
    return jnp.triu(jnp.full((q_len, k_len), jnp.finfo(jnp.float32).min, dtype=jnp.float32), 1)

class QwenMLP(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        self.intermediate_size = c.get("intermediate_size", 4 * self.hidden_size)
        self.gate_proj = nn.Dense(self.intermediate_size, dtype=self.dtype, use_bias=False, name="gate_proj")
        self.up_proj = nn.Dense(self.intermediate_size, dtype=self.dtype, use_bias=False, name="up_proj")
        self.down_proj = nn.Dense(self.hidden_size, dtype=self.dtype, use_bias=False, name="down_proj")
    def __call__(self, x):
        gate = jax.nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class QwenDecoderLayer(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        eps = c.get("rms_norm_eps", 1e-6)  # Checkpoint 6: consistent epsilon
        self.input_layernorm = nn.RMSNorm(epsilon=eps, dtype=self.dtype, name="input_layernorm")
        self.self_attn = QwenAttention(config=c, dtype=self.dtype)
        self.post_attention_layernorm = nn.RMSNorm(epsilon=eps, dtype=self.dtype, name="post_attention_layernorm")
        self.mlp = QwenMLP(config=c, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self attention
        batch, seq, _ = hidden_states.shape
        if position_ids is None:
            position_ids = jnp.arange(seq)[None, :].repeat(batch, axis=0)
        
        # Create attention mask for this layer
        if past_key_value is not None:
            past_k, _ = past_key_value
            key_len = past_k.shape[1] + seq
        else:
            key_len = seq
            
        # Create causal mask
        causal_mask = make_causal_mask(seq, key_len, dtype=self.dtype)
        causal_mask = causal_mask[None, None, :, :]  # [1, 1, seq, key_len]
        
        # Handle input attention mask
        if attention_mask is not None:
            if attention_mask.ndim == 2:  # [batch, seq]
                attention_mask = attention_mask[:, None, None, :]  # [batch, 1, 1, seq]
            # Convert to bias
            attention_bias = (1.0 - attention_mask.astype(jnp.float32)) * -1e9
            
            # Extend for cache if needed
            if key_len > seq:
                pad_len = key_len - seq
                pad_bias = jnp.zeros((batch, 1, 1, pad_len), dtype=jnp.float32)
                attention_bias = jnp.concatenate([pad_bias, attention_bias], axis=-1)
        else:
            attention_bias = jnp.zeros((batch, 1, 1, key_len), dtype=jnp.float32)
            
        # Combine causal mask and attention bias
        final_attention_mask = attention_bias + causal_mask
        
        cos, sin = compute_cos_sin_cache(position_ids, self.self_attn.head_dim, self.self_attn.rope_theta)
        
        hidden_states, past_key_value = self.self_attn(
            hidden_states,
            attention_mask=final_attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cos=cos,
            sin=sin
        )
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        
        return hidden_states, past_key_value

class Qwen25ForCausalLM(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    def setup(self):
        c = self.config
        self.vocab_size = c["vocab_size"]
        self.hidden_size = c["hidden_size"]
        self.num_layers = c["num_hidden_layers"]
        eps = c.get("rms_norm_eps", 1e-6)  # Checkpoint 6: consistent epsilon
        self.embed_tokens = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_size, dtype=self.dtype, name="embed_tokens")
        self.layers = [QwenDecoderLayer(config=c, dtype=self.dtype, name=f"layers_{i}") for i in range(self.num_layers)]
        self.norm = nn.RMSNorm(epsilon=eps, dtype=self.dtype, name="norm")
        self.lm_head = nn.Dense(self.vocab_size, dtype=self.dtype, use_bias=False, name="lm_head")

    def __call__(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, return_dict=True):
        batch, seq = input_ids.shape
        
        # Create position IDs if not provided
        if position_ids is None:
            if past_key_values is not None and past_key_values[0] is not None:
                past_length = past_key_values[0][0].shape[1]
                position_ids = jnp.arange(past_length, past_length + seq, dtype=jnp.int32)[None, :]
            else:
                position_ids = jnp.arange(seq, dtype=jnp.int32)[None, :]
        
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Initialize past key values if not provided
        if past_key_values is None:
            past_key_values = [None] * self.num_layers
        
        # Process each layer and collect new key-value caches
        new_key_values = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, new_key_value = layer(
                hidden_states, 
                attention_mask=attention_mask, 
                position_ids=position_ids,
                past_key_value=past_key_value
            )
            new_key_values.append(new_key_value)
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        if return_dict:
            return {
                "logits": logits,
                "past_key_values": new_key_values
            }
        return logits

# --- Weight loading (real mapping from run_inference.py/model.py) ---
def get_param_path(name):
    direct_mapping = {
        "model.embed_tokens.weight": ("embed_tokens", "embedding"),
        "model.norm.weight": ("norm", "scale"),
        "lm_head.weight": ("lm_head", "kernel"),
    }
    if name in direct_mapping:
        return direct_mapping[name]
    import re
    layer_norm_pattern = r"model\.layers\.(\d+)\.(input|post_attention)_layernorm\.weight"
    attention_pattern = r"model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.(weight|bias)"
    mlp_pattern = r"model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight"
    rotary_pattern = r"model\.layers\.(\d+)\.self_attn\.rotary_emb\..*"
    layer_norm_match = re.match(layer_norm_pattern, name)
    if layer_norm_match:
        layer_idx = int(layer_norm_match.group(1))
        norm_type = layer_norm_match.group(2)
        layer_name = f"layers_{layer_idx}"
        norm_name = "input_layernorm" if norm_type == "input" else "post_attention_layernorm"
        return (layer_name, norm_name, "scale")
    attn_match = re.match(attention_pattern, name)
    if attn_match:
        layer_idx = int(attn_match.group(1))
        proj_type = attn_match.group(2)
        param_type = attn_match.group(3)
        layer_name = f"layers_{layer_idx}"
        proj_name = f"{proj_type}_proj"
        param_name = "kernel" if param_type == "weight" else "bias"
        return (layer_name, "self_attn", proj_name, param_name)
    mlp_match = re.match(mlp_pattern, name)
    if mlp_match:
        layer_idx = int(mlp_match.group(1))
        proj_type = mlp_match.group(2)
        layer_name = f"layers_{layer_idx}"
        proj_name = f"{proj_type}_proj"
        return (layer_name, "mlp", proj_name, "kernel")
    rotary_match = re.match(rotary_pattern, name)
    if rotary_match:
        return None
    return None

def transpose_if_needed(name, param):
    """
    Correct transpose logic based on systematic debugging:
    - Projection layers (k_proj, v_proj, gate_proj, etc): NEED transpose 
    - lm_head: NO transpose (debug showed it worked before)
    - Embeddings, layer norms: NO transpose
    """
    if "embed_tokens.weight" in name:
        return param  # No transpose
    if "layernorm.weight" in name or "norm.weight" in name:
        return param  # No transpose for 1D weights
    if "lm_head.weight" in name:
        return param  # NO transpose for lm_head
    if "weight" in name and "proj" in name:
        return jnp.transpose(param)  # YES transpose for all projection layers
    return param

def process_safetensors_file(file_path, dtype=jnp.bfloat16):
    flax_params = {"params": {}}
    unmapped_keys = []
    
    with safe_open(file_path, framework="numpy") as f:
        for key in f.keys():
            param_path = get_param_path(key)
            if param_path is None:
                unmapped_keys.append(key)
                continue
                
            param = f.get_tensor(key)
            original_dtype = param.dtype
            original_shape = param.shape
            
            # Handle dtype conversion safely - separate weight and activation dtypes
            if original_dtype == np.float16:
                if "weight" in key and ("proj" in key or "lm_head" in key):
                    # Dense kernels: FP16 → FP32 → BF16 (weights in BF16)
                    param = param.astype(np.float32)
                    param = jnp.array(param, dtype=jnp.bfloat16)  # Weights always BF16
                else:
                    # Embeddings and LayerNorm: keep in FP16 for embeddings, BF16 for others
                    target_dtype = jnp.float16 if "embed_tokens" in key else jnp.bfloat16
                    param = jnp.array(param, dtype=target_dtype)
            else:
                # For non-FP16 inputs, convert appropriately
                if "weight" in key and ("proj" in key or "lm_head" in key):
                    param = jnp.array(param, dtype=jnp.bfloat16)  # Weights in BF16
                else:
                    param = jnp.array(param, dtype=jnp.bfloat16)
            
            param_before_transpose = param
            param = transpose_if_needed(key, param)
            
            # Validate transpose worked as expected
            if "weight" in key and ("proj" in key or "lm_head" in key):
                if jnp.array_equal(param, param_before_transpose):
                    logger.warning(f"Expected transpose for {key} but array unchanged")
                else:
                    # Quick checksum to catch double-transpose
                    before_mean = jnp.mean(param_before_transpose[:min(2, param_before_transpose.shape[0]), :min(2, param_before_transpose.shape[1])])
                    after_mean = jnp.mean(param[:min(2, param.shape[0]), :min(2, param.shape[1])])
                    logger.debug(f"Transpose {key}: before_mean={float(before_mean):.6f}, after_mean={float(after_mean):.6f}")
            
            current_dict = flax_params["params"]
            for path_part in param_path[:-1]:
                if path_part not in current_dict:
                    current_dict[path_part] = {}
                current_dict = current_dict[path_part]
            current_dict[param_path[-1]] = param
            
            logger.debug(f"Loaded {key} -> {'/'.join(param_path)}: {original_shape} {original_dtype} -> {param.shape} {param.dtype}")
            del param
            gc.collect()
    
    if unmapped_keys:
        logger.info(f"Unmapped keys in {os.path.basename(file_path)}: {unmapped_keys}")
    
    return flax_params

def merge_param_dicts(base_dict, new_dict):
    for key, value in new_dict.items():
        if key not in base_dict:
            base_dict[key] = value
        elif isinstance(value, dict) and isinstance(base_dict[key], dict):
            merge_param_dicts(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def load_params(model, model_path, dtype):
    """Load model parameters from safetensors files."""
    logger.info("Loading weights...")
    
    # 1. Initialize full param tree with dummy input
    dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
    init_params = model.init(jax.random.PRNGKey(0), dummy_input)
    
    # 2. Load weights from safetensors files
    param_dict = {}
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            file_path = os.path.join(model_path, file)
            logger.info(f"Loading {file}")
            file_params = process_safetensors_file(file_path, dtype)
            param_dict = merge_param_dicts(param_dict, file_params)
    
    # 3. Map weights to model structure
    def map_params(params, param_dict):
        if isinstance(params, dict):
            out = {}
            for k, v in params.items():
                if k in param_dict:  # <- use loaded value if present
                    out[k] = map_params(v, param_dict[k])
                else:
                    out[k] = v
            return out
        else:  # leaf – replace if we have it
            return param_dict if isinstance(param_dict, (jnp.ndarray, np.ndarray)) else params
    
    # 4. Update initialized params with loaded weights
    params = map_params(init_params, param_dict)
    
    # 5. Validation checks
    logger.info("Validating loaded weights...")
    
    # Check that weights actually changed from initialization
    init_embed_std = jnp.std(init_params['params']['embed_tokens']['embedding'])
    loaded_embed_std = jnp.std(params['params']['embed_tokens']['embedding'])
    logger.info(f"Embedding std - init: {float(init_embed_std):.6f}, loaded: {float(loaded_embed_std):.6f}")
    
    if abs(float(init_embed_std) - float(loaded_embed_std)) < 1e-6:
        logger.warning("WARNING: Embedding weights appear unchanged from initialization!")
    
    # Count total parameters
    def count_params(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return sum(np.prod(leaf.shape) for leaf in leaves)
    
    total_params = count_params(params)
    logger.info(f"Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    
    # Check if embed_tokens and lm_head are tied (should be for Qwen 2.5)
    embed_tokens = params['params']['embed_tokens']['embedding']
    lm_head = params['params']['lm_head']['kernel']
    
    if embed_tokens.shape == lm_head.shape:
        max_diff = jnp.max(jnp.abs(embed_tokens - lm_head))
        logger.info(f"Embed↔LM-head tie check: max_diff = {float(max_diff):.2e}")
        if float(max_diff) < 1e-6:
            logger.info("✓ Weights are properly tied")
        else:
            logger.warning("✗ Weights are NOT tied (this may be expected)")
    else:
        logger.info(f"Embed shape: {embed_tokens.shape}, LM head shape: {lm_head.shape}")
    
    return params

# --- Chat template support ---
def apply_chat_template(tokenizer, messages):
    """Apply the Qwen chat template to messages."""
    # Use the tokenizer's built-in chat template - no fallback (checkpoint 2)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

# --- Generation (with all improvements) ---
def apply_repetition_penalty(logits, past_ids, penalty):
    """Apply sign-aware repetition penalty - dtype-safe & sign aware."""
    if penalty == 1.0 or not past_ids:
        return logits
    
    for tok in past_ids[-50:]:  # Only consider last 50 tokens
        if 0 <= tok < logits.shape[-1]:
            tok_slice = logits[..., tok]
            logits = logits.at[..., tok].set(
                jnp.where(tok_slice >= 0,
                          tok_slice / penalty,
                          tok_slice * penalty)
            )
    return logits

def sample_next_token(logits, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05, past_tokens=None, rng_key=None):
    """Sample from logits with official Qwen2.5-7B-Instruct parameters."""
    
    # Apply sign-aware repetition penalty if we have past tokens
    if repetition_penalty != 1.0 and past_tokens is not None and len(past_tokens) > 0:
        logits = apply_repetition_penalty(logits, past_tokens, repetition_penalty)
    
    # Greedy branch - bypass all sampling when temperature == 0
    if temperature == 0.0:
        return jnp.argmax(logits, axis=-1)
    
    # Apply temperature
    logits = logits / temperature
    
    # Top-k filtering (official: k=20)
    if top_k > 0:
        # Handle batch dimension properly
        batch_size = logits.shape[0]
        vocab_size = logits.shape[-1]
        
        # Get top-k for each batch element
        top_k_values, top_k_indices = jax.lax.top_k(logits, k=min(top_k, vocab_size))
        
        # Create mask for top-k tokens
        mask = jnp.full_like(logits, False, dtype=bool)
        for b in range(batch_size):
            mask = mask.at[b, top_k_indices[b]].set(True)
        
        # Set non-top-k logits to -inf
        logits = jnp.where(mask, logits, -jnp.inf)
    
    # Top-p (nucleus) filtering (official: p=0.8)
    if top_p < 1.0:
        # Sort logits in descending order
        sorted_indices = jnp.argsort(logits, axis=-1)[..., ::-1]
        sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
        sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
        cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
        
        # Find indices to remove (cumulative prob > top_p)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least the first token
        sorted_indices_to_remove = sorted_indices_to_remove.at[..., 0].set(False)
        
        # Map back to original indices
        indices_to_remove = jnp.zeros_like(logits, dtype=bool)
        for b in range(logits.shape[0]):
            indices_to_remove = indices_to_remove.at[b, sorted_indices[b]].set(sorted_indices_to_remove[b])
        
        logits = jnp.where(indices_to_remove, -jnp.inf, logits)
    
    # Sample from the filtered distribution using provided RNG key
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    return jax.random.categorical(rng_key, logits, axis=-1)

def generate_text(model, params, tokenizer, prompt, max_tokens, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05, use_chat_template=True, seed=0):
    """Generate text using the model with official parameters and proper EOS handling."""
    
    # Configure tokenizer with both EOS tokens (checkpoint 3)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")      # 151645
    endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")      # 151643
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.eos_token_id = im_end_id
    STOP_IDS = {im_end_id, endoftext_id}  # Stop on both tokens (checkpoint 3)
    
    # Apply chat template if requested
    if use_chat_template:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = apply_chat_template(tokenizer, messages)
        logger.info(f"Chat template applied. Formatted prompt: {repr(formatted_prompt[:200])}...")
    else:
        formatted_prompt = prompt
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    
    # Create attention mask - use 2D format, model will convert to 4D
    batch_size = input_ids.shape[0]
    seq_length = input_ids.shape[1]
    attention_mask = np.ones((batch_size, seq_length), dtype=np.int32)
    
    # Position IDs - make sure to match the input_ids length
    position_ids = np.arange(input_ids.shape[1], dtype=np.int32)[None, :]
    
    # Initialize generation state
    state = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": None,
    }
    
    # Track generated text and tokens for repetition penalty
    generated_text = ""
    generated_tokens = []
    
    # Create one RNG before the loop and split it each step (checkpoint 8)
    rng = jax.random.PRNGKey(seed)
    
    logger.info(f"Starting generation with {input_ids.shape[1]} input tokens...")
    logger.info(f"Using official parameters: temperature={temperature}, top_p={top_p}, top_k={top_k}, repetition_penalty={repetition_penalty}")
    logger.info(f"Stop tokens: <|im_end|> ({im_end_id}) and <|endoftext|> ({endoftext_id})")
    
    # Generate tokens
    for i in range(max_tokens):
        # Forward pass
        outputs = model.apply(
            params,
            input_ids=state["input_ids"],
            attention_mask=state["attention_mask"],
            position_ids=state["position_ids"],
            past_key_values=state["past_key_values"],
            return_dict=True
        )
        
        # Get logits and past key values
        logits = outputs["logits"]
        past_key_values = outputs["past_key_values"]
        
        # Split RNG each step (checkpoint 8)
        rng, subkey = jax.random.split(rng)
        
        # Sample next token with repetition penalty
        next_token = sample_next_token(
            logits[:, -1, :], 
            temperature=temperature, 
            top_p=top_p, 
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            past_tokens=generated_tokens,
            rng_key=subkey
        )
        
        # Track generated token for repetition penalty
        generated_tokens.append(int(next_token[0]))
        
        # Update state for next iteration
        state["input_ids"] = next_token[:, None]
        state["attention_mask"] = np.ones((batch_size, 1), dtype=np.int32)  # Single token mask
        state["position_ids"] = np.array([[state["position_ids"][0, -1] + 1]], dtype=np.int32)
        state["past_key_values"] = past_key_values
        
        # Decode and print token with trim whitespace (checkpoint 3)
        token_str = tokenizer.decode(int(next_token[0]), skip_special_tokens=True).rstrip()
        generated_text += token_str
        print(token_str, end="", flush=True)
        
        # Check for end of sequence - stop on both EOS tokens (checkpoint 3)
        if int(next_token[0]) in STOP_IDS:
            token_name = "<|im_end|>" if int(next_token[0]) == im_end_id else "<|endoftext|>"
            logger.info(f"\nStopped at {token_name} after {i+1} tokens")
            break
    
    print()  # New line at end
    return generated_text

# --- Layer-by-layer parity test (checkpoint 10) ---
def test_parity_with_pytorch(model, params, model_path):
    """Test layer-by-layer parity with PyTorch implementation."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        prompt = "1 2 3 ++"
        ref_tok = AutoTokenizer.from_pretrained(model_path)
        pt = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        
        text = ref_tok.apply_chat_template(
            [{"role": "user", "content": prompt}], 
            tokenize=False, 
            add_generation_prompt=True
        )
        ids = ref_tok(text, return_tensors="pt").input_ids
        
        with torch.no_grad():
            pt_logits = pt(ids).logits.numpy()
        
        flax_logits = model.apply(params, jnp.array(ids.numpy()))["logits"]
        diff = np.abs(pt_logits - flax_logits).max()
        
        logger.info(f"Layer-by-layer parity test:")
        logger.info(f"Input: {repr(prompt)}")
        logger.info(f"PyTorch logits shape: {pt_logits.shape}")
        logger.info(f"JAX logits shape: {flax_logits.shape}")
        logger.info(f"Max absolute difference: {diff}")
        
        if diff < 1e-5:
            logger.info("✓ PARITY TEST PASSED - Models are bit-for-bit identical!")
        else:
            logger.warning(f"✗ PARITY TEST FAILED - Difference {diff} exceeds threshold")
        
        return diff < 1e-5
        
    except ImportError:
        logger.warning("PyTorch/transformers not available for parity test")
        return None
    except Exception as e:
        logger.error(f"Parity test failed: {e}")
        return False

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B-Instruct Inference (fixed for mathematical reasoning)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (official: 0.7)")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p sampling parameter (official: 0.8)")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling parameter (official: 20)")
    parser.add_argument("--repetition_penalty", type=float, default=1.05, help="Repetition penalty (official: 1.05)")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"], help="Activation dtype (default: float32 for BF16-weights/FP32-activations)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generation")
    parser.add_argument("--no_chat_template", action="store_true", help="Don't apply chat template")
    parser.add_argument("--test_parity", action="store_true", help="Run layer-by-layer parity test with PyTorch")
    args = parser.parse_args()
    
    # Use FP32 for activations (BF16-weights/FP32-activations mode)
    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    
    # Load config
    config_path = os.path.join(args.model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Loaded config: {config}")
    logger.info(f"Using activation dtype: {dtype} (weights will be BF16)")
    model = Qwen25ForCausalLM(config=config, dtype=dtype)
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load weights (weights in BF16, activations in specified dtype)
    params = load_params(model, args.model_path, dtype)
    gc.collect(); jax.clear_caches()
    
    # Run parity test if requested
    if args.test_parity:
        logger.info("Running layer-by-layer parity test...")
        parity_result = test_parity_with_pytorch(model, params, args.model_path)
        if parity_result is True:
            logger.info("Parity test passed! Proceeding with generation...")
        elif parity_result is False:
            logger.warning("Parity test failed! Generation may produce different results than PyTorch.")
        else:
            logger.info("Parity test skipped (PyTorch not available).")
    
    # Generate with official parameters and proper EOS handling
    generate_text(
        model, params, tokenizer, args.prompt, args.max_tokens, 
        args.temperature, args.top_p, args.top_k, args.repetition_penalty,
        use_chat_template=not args.no_chat_template, seed=args.seed
    )
    
    # Clean up
    del params; del model; gc.collect(); jax.clear_caches()

if __name__ == "__main__":
    main() 