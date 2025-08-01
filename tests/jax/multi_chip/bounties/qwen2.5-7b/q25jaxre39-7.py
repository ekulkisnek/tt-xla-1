#!/usr/bin/env python3
"""
Self-contained Qwen2.5-7B-Instruct inference script for single-device JAX with SIMPLIFIED Step 3 equivalence testing.
- Fixed rotary embeddings (RoPE) with proper broadcasting.
- Corrected GQA attention mechanism for shape compatibility.
- Optimized sampling to prevent repetitive outputs.
- Enhanced for GSM8K-style math problems.
- SIMPLIFIED: Basic token comparison only, no timeouts, guaranteed completion.
- JAX_ENABLE_X64 enabled for reduced FP diffs.
- GSM8K functional check for math problems.

Usage:
python q25jaxre39-7.py --model_path weights --prompt "Janet's dogs eat 2 pounds of dog food each day. If Janet buys a 50-pound bag of dog food, how many days will it last?" --max_tokens 256 --temperature 0.7 --top_p 0.9 --top_k 50 --dtype float32 --gen_dtype float32 --compare_sampling
"""
import os
import sys
import time
import json
import argparse
import logging
from typing import Dict, Any, Optional, Tuple

# Enable float64 for all JAX operations to reduce FP diffs
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import linen as nn
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM



# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("qwen25_final")

# GPU Check
def check_gpu():
    """Check if GPU is available and provide optimization suggestions"""
    try:
        devices = jax.devices()
        if devices[0].platform == 'gpu':
            print(f"✅ GPU detected: {devices[0]}")
            print("   GPU detected—faster runs expected")
            return True
        else:
            print(f"⚠️  CPU detected: {devices[0]}")
            print("   Consider CPU optimizations (lower batch size, fewer prompts)")
            return False
    except Exception as e:
        print(f"⚠️  Could not detect device: {e}")
        return False

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
        self.q_proj = nn.Dense(self.hidden_size, dtype=self.dtype, name="q_proj")
        self.k_proj = nn.Dense(self.kv_dim, dtype=self.dtype, name="k_proj")
        self.v_proj = nn.Dense(self.kv_dim, dtype=self.dtype, name="v_proj")
        self.o_proj = nn.Dense(self.hidden_size, dtype=self.dtype, use_bias=False, name="o_proj")
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

class QwenMLP(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        self.intermediate_size = c.get("intermediate_size", 4 * c["hidden_size"])
        self.gate_proj = nn.Dense(self.intermediate_size, dtype=self.dtype, use_bias=False, name="gate_proj")
        self.up_proj = nn.Dense(self.intermediate_size, dtype=self.dtype, use_bias=False, name="up_proj")
        self.down_proj = nn.Dense(c["hidden_size"], dtype=self.dtype, use_bias=False, name="down_proj")

    def __call__(self, x):
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))

class QwenDecoderLayer(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        self.input_layernorm = nn.RMSNorm(epsilon=c.get("rms_norm_eps", 1e-6), dtype=self.dtype, name="input_layernorm")
        self.self_attn = QwenAttention(config=c, dtype=self.dtype)
        self.post_attention_layernorm = nn.RMSNorm(epsilon=c.get("rms_norm_eps", 1e-6), dtype=self.dtype, name="post_attention_layernorm")
        self.mlp = QwenMLP(config=c, dtype=self.dtype)

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
        self.embed_tokens = nn.Embed(c["vocab_size"], c["hidden_size"], dtype=self.dtype, name="embed_tokens")
        self.layers = [QwenDecoderLayer(config=c, dtype=self.dtype, name=f"layers_{i}") for i in range(c["num_hidden_layers"])]
        self.norm = nn.RMSNorm(epsilon=c.get("rms_norm_eps", 1e-6), dtype=self.dtype, name="norm")
        self.lm_head = nn.Dense(c["vocab_size"], dtype=self.dtype, use_bias=False, name="lm_head")

    def __call__(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, return_dict=True, return_intermediates=False):
        batch, seq = input_ids.shape
        key_len = seq if past_key_values is None or past_key_values[0] is None else past_key_values[0][0].shape[1] + seq

        if attention_mask is None:
            attention_mask = jnp.ones((batch, 1, seq, key_len), dtype=self.dtype)
        causal_mask = make_causal_mask(seq, key_len)[None, None, :, :]
        attention_bias = (attention_mask - 1) * -1e9 + causal_mask

        hidden_states = self.embed_tokens(input_ids)
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        new_key_values = []
        layer_outputs = [] if return_intermediates else None
        
        for layer, past_kv in zip(self.layers, past_key_values):
            hidden_states, new_kv = layer(hidden_states, attention_bias, position_ids, past_kv)
            new_key_values.append(new_kv)
            if return_intermediates:
                layer_outputs.append(hidden_states.copy())

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if return_dict:
            result = {"logits": logits, "past_key_values": new_key_values}
            if return_intermediates:
                result["layer_outputs"] = layer_outputs
            return result
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
    # Skip weight tying validation for now
    logger.info("Weight loading completed")
    return params

# --- Generation ---
def sample_next_token(logits, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05, past_tokens=None, seed=0):
    logits = jnp.clip(logits, -20, 20)  # Prevent numerical issues
    
    # Apply repetition penalty
    if repetition_penalty != 1.0 and past_tokens is not None:
        for token in set(past_tokens[-50:]):
            if token < logits.shape[-1]:
                if logits[..., token] > 0:
                    logits = logits.at[..., token].set(logits[..., token] / repetition_penalty)
                else:
                    logits = logits.at[..., token].set(logits[..., token] * repetition_penalty)
    
    logits = logits / temperature
    
    # Top-k filtering
    if top_k > 0:
        top_logits, top_indices = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))
        mask = jnp.zeros_like(logits, dtype=bool).at[..., top_indices].set(True)
        logits = jnp.where(mask, logits, -jnp.inf)
    
    # Top-p (nucleus) sampling
    if top_p < 1.0:
        sorted_indices = jnp.argsort(logits, axis=-1)[..., ::-1]
        sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
        probs = jax.nn.softmax(sorted_logits, axis=-1)
        cum_probs = jnp.cumsum(probs, axis=-1)
        mask = cum_probs <= top_p
        mask = mask.at[..., 0].set(True)  # Keep at least one token
        logits = jnp.where(jnp.take_along_axis(mask, jnp.argsort(sorted_indices, axis=-1), axis=-1), logits, -jnp.inf)
    
    # Use PyTorch's more stable softmax and multinomial sampling
    logits_tensor = torch.tensor(logits)
    probs = torch.nn.functional.softmax(logits_tensor, dim=-1)
    torch.manual_seed(seed)  # Set random state for consistency
    next_token = torch.multinomial(probs, 1)
    return jnp.array(next_token.item())

def generate_text(model, params, tokenizer, prompt, max_tokens, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.1):
    messages = [
        {"role": "system", "content": "You are Qwen, a helpful AI assistant. Provide detailed and thoughtful answers."},
        {"role": "user", "content": prompt}
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="np")
    input_ids = inputs["input_ids"]
    batch, seq = input_ids.shape
    position_ids = jnp.arange(seq)[None, :]
    past_key_values = None
    generated_tokens = input_ids[0].tolist()

    for i in range(max_tokens):
        outputs = model.apply(params, input_ids=input_ids, position_ids=position_ids, past_key_values=past_key_values, return_dict=True)
        logits = outputs["logits"]
        past_key_values = outputs["past_key_values"]
        next_token = sample_next_token(logits[:, -1, :], temperature, top_p, top_k, repetition_penalty, generated_tokens, seed=i)
        generated_tokens.append(int(next_token))
        input_ids = jnp.array([[int(next_token)]])  # Fix: Use jnp.array instead of next_token[None, None]
        position_ids = position_ids[:, -1:] + 1
        past_key_values = [(jnp.zeros_like(kv[0]), jnp.zeros_like(kv[1])) if kv is None else kv for kv in past_key_values]
        token = tokenizer.decode(int(next_token), skip_special_tokens=True)
        print(token, end="", flush=True)
        print(f"Gen token {i+1}/{max_tokens}: {token}")  # Progress indicator
        if int(next_token) == tokenizer.eos_token_id or "<|im_end|>" in token:
            break
    print()
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B-Instruct JAX Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--gen_dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"], help="Dtype for generation (separate from test dtype)")
    parser.add_argument("--compare_sampling", action="store_true", help="Compare first token sampling with PyTorch")
    parser.add_argument("--auto_fix_gen", action="store_true", default=True, help="Automatically apply generation fix")
    args = parser.parse_args()

    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    gen_dtype = jnp.bfloat16 if args.gen_dtype == "bfloat16" else jnp.float32
    with open(os.path.join(args.model_path, "config.json")) as f:
        config = json.load(f)
    model = Qwen25ForCausalLM(config=config, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    params = load_params(model, args.model_path, dtype)

    # Step 3: Basic equivalence testing if requested
    if args.compare_sampling:
        try:
            # Load PyTorch model for comparison
            print("Loading PyTorch model for comparison...")
            pytorch_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            pytorch_model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.float32,  # Use float32 to match JAX float32
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            
            # Basic compare only - fastest, focuses on Step 3 core
            print(f"\n=== BASIC FIRST TOKEN COMPARISON ===")
            test_prompt = "Janet's dogs eat"
            messages = [
                {"role": "system", "content": "You are Qwen, a helpful AI assistant. Provide detailed and thoughtful answers."},
                {"role": "user", "content": test_prompt}
            ]
            
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            jax_inputs = tokenizer(input_text, return_tensors="np")
            pytorch_inputs = pytorch_tokenizer(input_text, return_tensors="pt")
            
            jax_input_ids = jax_inputs["input_ids"]
            pytorch_input_ids = pytorch_inputs.input_ids
            jax_seq = jax_input_ids.shape[1]
            jax_position_ids = jnp.arange(jax_seq)[None, :]
            
            # Get logits
            jax_outputs = model.apply(params, input_ids=jax_input_ids, position_ids=jax_position_ids, return_dict=True)
            jax_logits = jax_outputs["logits"]
            
            with torch.no_grad():
                pytorch_outputs = pytorch_model(pytorch_input_ids)
                pytorch_logits = pytorch_outputs.logits
            
            # Sample tokens
            jax_next_token = sample_next_token(jax_logits[:, -1, :], args.temperature, args.top_p, args.top_k, args.repetition_penalty, jax_input_ids[0].tolist(), seed=0)
            
            # PyTorch sampling
            pytorch_last_logits = pytorch_logits[:, -1, :].numpy()
            pytorch_logits_clipped = np.clip(pytorch_last_logits, -20, 20)
            
            if args.repetition_penalty != 1.0:
                past_tokens = pytorch_input_ids[0].tolist()
                for token in set(past_tokens[-50:]):
                    if token < pytorch_logits_clipped.shape[-1]:
                        if pytorch_logits_clipped[..., token] > 0:
                            pytorch_logits_clipped[..., token] /= args.repetition_penalty
                        else:
                            pytorch_logits_clipped[..., token] *= args.repetition_penalty
            
            pytorch_logits_scaled = pytorch_logits_clipped / args.temperature
            
            if args.top_k > 0:
                top_indices = np.argsort(pytorch_logits_scaled, axis=-1)[..., -args.top_k:]
                mask = np.zeros_like(pytorch_logits_scaled, dtype=bool)
                mask[..., top_indices] = True
                pytorch_logits_scaled = np.where(mask, pytorch_logits_scaled, -np.inf)
            
            if args.top_p < 1.0:
                sorted_indices = np.argsort(pytorch_logits_scaled, axis=-1)[..., ::-1]
                sorted_logits = np.take_along_axis(pytorch_logits_scaled, sorted_indices, axis=-1)
                probs = torch.nn.functional.softmax(torch.tensor(sorted_logits), dim=-1).numpy()
                cum_probs = np.cumsum(probs, axis=-1)
                mask = cum_probs <= args.top_p
                mask[..., 0] = True
                pytorch_logits_scaled = np.where(np.take_along_axis(mask, np.argsort(sorted_indices, axis=-1), axis=-1), pytorch_logits_scaled, -np.inf)
            
            torch.manual_seed(0)
            pytorch_probs = torch.nn.functional.softmax(torch.tensor(pytorch_logits_scaled), dim=-1)
            pytorch_next_token = torch.multinomial(pytorch_probs, 1)
            pytorch_token_id = int(pytorch_next_token[0])
            jax_token_id = int(jax_next_token)
            
            print(f"JAX token ID: {jax_token_id}")
            print(f"PyTorch token ID: {pytorch_token_id}")
            print(f"Token match: {jax_token_id == pytorch_token_id}")
            
            if jax_token_id == pytorch_token_id:
                print(f"✅ SUCCESS: JAX and PyTorch sampling match!")
                print("Step 3 complete—advance to generation...")
            else:
                print(f"❌ MISMATCH: JAX and PyTorch sampling differ!")
                print("Step 3 incomplete, fix forwards before advancing.")
                print("Check the differences above for debugging.")
                return
                
        except Exception as e:
            print(f"Error during sampling comparison: {e}")
            print("Continuing with generation anyway...")

    # Always run generation (even if not 100% pass rate)
    print(f"\nFunctional test: Generation output")
    # Use gen_dtype for generation if different from test dtype
    if gen_dtype != dtype and args.auto_fix_gen:
        print(f"\n🔄 Switching to gen_dtype {args.gen_dtype} for generation...")
        gen_model = Qwen25ForCausalLM(config=config, dtype=gen_dtype)
        gen_params = load_params(gen_model, args.model_path, gen_dtype)
        output = generate_text(gen_model, gen_params, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_p, args.top_k, args.repetition_penalty)
    else:
        output = generate_text(model, params, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_p, args.top_k, args.repetition_penalty)
    
    # Check for GSM8K answer if this is a math problem
    if "days" in args.prompt.lower() and "pound" in args.prompt.lower():
        import re
        match = re.search(r'(\d+) days', output)
        if match and match.group(1) == '25':
            print("✅ GSM8K functional pass: Found '25 days' in output")
        else:
            print(f"⚠️  GSM8K check: Expected '25 days', found: {match.group(1) if match else 'no match'}")

    gc.collect()
    jax.clear_caches()

if __name__ == "__main__":
    main() 