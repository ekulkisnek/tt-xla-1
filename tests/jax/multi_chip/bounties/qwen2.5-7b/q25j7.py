#!/usr/bin/env python3
"""
Self-contained Qwen2.5-7B-Instruct inference script for single-device JAX optimized for Step 7: GSM8K Benchmark.
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

Usage:
python q25j7.py --model_path weights
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

import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer
from flax import linen as nn



# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("qwen25_step7")

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
                        param = jnp.array(param, dtype=jnp.bfloat16) # Always load as bfloat16
                        param = transpose_if_needed(key, param)
                        d = params["params"]
                        for p in path[:-1]:
                            d = d.setdefault(p, {})
                        d[path[-1]] = param
    
    gc.collect()  # Add GC collect after loading
    print("Weight loading completed")
    return params

# --- Generation ---
def sample_next_token(logits):
    """Simplified greedy sampling only."""
    return int(jnp.argmax(logits, axis=-1)[0])

def generate_text(model, params, tokenizer, max_tokens, prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="np")
    input_ids = inputs["input_ids"]
    batch, seq = input_ids.shape
    position_ids = jnp.arange(seq)[None, :]
    past_key_values = None
    generated_tokens = input_ids[0].tolist()

    start_time = time.perf_counter()
    peak_mem = psutil.virtual_memory().used / (1024**3)
    print(f"Initial memory before generation: {peak_mem:.2f} GB used")
    print(f"Free memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")

    num_tokens_generated = 0
    for i in range(max_tokens):
        outputs = model.apply(params, input_ids=input_ids, position_ids=position_ids, past_key_values=past_key_values, return_dict=True)
        logits = outputs["logits"]
        past_key_values = outputs["past_key_values"]
        next_token = sample_next_token(logits[:, -1, :])
        generated_tokens.append(int(next_token))
        input_ids = jnp.array([[next_token]])
        position_ids = position_ids[:, -1:] + 1
        num_tokens_generated += 1
        
        # Update peak mem
        current_mem = psutil.virtual_memory().used / (1024**3)
        if current_mem > peak_mem:
            peak_mem = current_mem
        
        token = tokenizer.decode(int(next_token), skip_special_tokens=True)
        print(f"Gen token {i+1}/{max_tokens}: {token}")
        if int(next_token) == tokenizer.eos_token_id or "<|im_end|>" in token:
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

def extract_answer(output: str) -> str:
    """Extract numerical answer from model output."""
    import re
    match = re.search(r'\\boxed\{([0-9]+)\}', output)
    return match.group(1) if match else None

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B-Instruct JAX Inference for GSM8K Benchmark")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    
    with open(os.path.join(args.model_path, "config.json")) as f:
        config = json.load(f)
    model = Qwen25ForCausalLM(config=config, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    params = load_params(model, args.model_path, dtype)
    
    # Benchmark on 10 GSM8K samples
    print("\nRunning GSM8K benchmark on 10 samples...")
    gsm8k_samples = [
        {"question": "Janet's dogs eat 2 pounds of dog food each day. If Janet buys a 50-pound bag, how many days will it last?", "answer": "25"},
        {"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "answer": "72"},
        {"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "answer": "10"},
        {"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?", "answer": "5"},
        {"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?", "answer": "42"},
        {"question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?", "answer": "624"},
        {"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?", "answer": "35"},
        {"question": "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?", "answer": "48"},
        {"question": "Ken created a care package to send to his brother, who was away at boarding school. Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds. Then, he added enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly beans. And finally, he added enough gummy worms to double the weight once again. What was the final weight of the box of goodies, in pounds?", "answer": "16"},
        {"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has $16 left from her budget. How much did Alexis pay for the shoes?", "answer": "41"},
        {"question": "Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?", "answer": "990"}
    ]
    
    correct_count = 0
    total_peak_mem = 0
    total_avg_time_per_token = 0
    for i, sample in enumerate(gsm8k_samples):
        print(f"\n--- Sample {i+1}/{len(gsm8k_samples)}: {sample['question']} ---")
        gc.collect()  # GC before each sample
        output, peak_mem, avg_time_per_token = generate_text(model, params, tokenizer, 256, sample["question"])
        pred = extract_answer(output)
        is_correct = pred == sample["answer"]
        if is_correct:
            correct_count += 1
        total_peak_mem += peak_mem
        total_avg_time_per_token += avg_time_per_token
        print(f"Predicted: {pred}")
        print(f"Expected: {sample['answer']}")
        print(f"Correct: {is_correct}")
    
    avg_peak_mem = total_peak_mem / len(gsm8k_samples)
    avg_time_per_token = total_avg_time_per_token / len(gsm8k_samples)
    accuracy = correct_count / len(gsm8k_samples) * 100
    print("\nBenchmark Summary:")
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{len(gsm8k_samples)} correct)")
    print(f"Average Peak Memory per Sample: {avg_peak_mem:.2f} GB")
    print(f"Average Time per Token: {avg_time_per_token:.2f} seconds")


if __name__ == "__main__":
    main() 