#!/usr/bin/env python3
"""
Qwen2.5-7B-Instruct JAX Inference (gpt5 variant) with tensor parallelism and architecture toggles.

Goal: eliminate early repetition (e.g., "needs needs") seen in q25p-8-math-final-fixed.py by aligning
core architectural parts to the known-good implementation (q25psammath.py) and exposing switches to
isolate the exact root cause.

Defaults mirror the known-good path:
- RoPE: cos/sin method (use --rope complex to test complex precompute variant)
- KV cache: explicit via past_key_values only (no internal mutable cache)
- Embeddings: ParallelEmbed
- LM head: ParallelDense (use --lm_head dense to test nn.Dense)

This script keeps deterministic greedy decoding and a simple generation loop.
"""
import os
import json
import argparse
import psutil
import gc
import time
import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple
from safetensors import safe_open
from flax import linen as nn
from transformers import AutoTokenizer
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

# Environment
os.environ.setdefault("JAX_ENABLE_X64", "0")
os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=8')

# Global mesh for tensor parallelism
mesh = None

# -------- RoPE utilities ---------
def compute_cos_sin_cache(position_ids, head_dim, rope_theta=1000000.0):
    pos = position_ids.astype(jnp.float32)  # [batch, seq]
    dim = head_dim // 2
    freqs = 1.0 / (rope_theta ** (jnp.arange(0, dim, dtype=jnp.float32) / dim))
    t = pos[..., None] * freqs[None, None, :]
    cos = jnp.cos(t)
    sin = jnp.sin(t)
    cos = cos[..., None, :]
    sin = sin[..., None, :]
    return cos, sin


def apply_rotary_emb(q, k, cos, sin):
    half_dim = q.shape[-1] // 2
    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    k1, k2 = k[..., :half_dim], k[..., half_dim:]
    q_rot = jnp.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
    k_rot = jnp.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)
    return q_rot, k_rot


def precompute_freqs_cis(dim: int, end: int, theta: float = 1000000.0):
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
    t = jnp.arange(end, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs).astype(jnp.float32)
    return jnp.exp(1j * freqs)


def apply_rotary_emb_complex(q, k, freqs_cis):
    half_dim = q.shape[-1] // 2
    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    k1, k2 = k[..., :half_dim], k[..., half_dim:]
    q_complex = jax.lax.complex(q1.astype(jnp.float32), q2.astype(jnp.float32))
    k_complex = jax.lax.complex(k1.astype(jnp.float32), k2.astype(jnp.float32))
    freqs_cis_expanded = freqs_cis[..., None, :]
    q_rot = q_complex * freqs_cis_expanded
    k_rot = k_complex * freqs_cis_expanded
    q_rot_real = jnp.concatenate([jnp.real(q_rot), jnp.imag(q_rot)], axis=-1)
    k_rot_real = jnp.concatenate([jnp.real(k_rot), jnp.imag(k_rot)], axis=-1)
    return q_rot_real.astype(q.dtype), k_rot_real.astype(k.dtype)


# -------- Core blocks ---------
class ParallelEmbed(nn.Module):
    num_embeddings: int
    features: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    name: str = None

    def setup(self):
        self.embedding = self.param(
            "embedding",
            nn.initializers.normal(stddev=0.02),
            (self.num_embeddings, self.features),
            self.param_dtype,
        )

    def __call__(self, inputs):
        embedding = jnp.asarray(self.embedding, self.dtype)
        return embedding[inputs.astype("i4")]


class StandardEmbed(nn.Module):
    num_embeddings: int
    features: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    name: str = None

    def setup(self):
        self.embedding = self.param(
            "embedding",
            nn.initializers.normal(stddev=0.02),
            (self.num_embeddings, self.features),
            self.param_dtype,
        )

    def __call__(self, inputs):
        embedding = jnp.asarray(self.embedding, self.dtype)
        return embedding[inputs.astype("i4")]


class ParallelDense(nn.Module):
    features: int
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    use_bias: bool = False
    name: str = None

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        in_dim = x.shape[-1]
        out_dim = self.features
        kernel = self.param(
            "kernel", nn.initializers.lecun_normal(), (in_dim, out_dim), self.param_dtype
        )
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (out_dim,), self.param_dtype)
        else:
            bias = None

        def matmul_fn(x_local, k_local, b_local=None):
            local_out = jnp.einsum("bsd,df->bsf", x_local, k_local)
            if b_local is not None:
                local_out = local_out + b_local
            full_out = jax.lax.all_gather(local_out, axis_name="mp", axis=0)
            result = jnp.reshape(jnp.transpose(full_out, (1, 2, 0, 3)), (x_local.shape[0], x_local.shape[1], -1))
            return result

        if bias is not None:
            output = shard_map(
                matmul_fn,
                mesh=mesh,
                in_specs=(None, P(None, "mp"), P("mp",)),
                out_specs=P(None),
                check_rep=False,
            )(x, kernel, bias)
        else:
            output = shard_map(
                matmul_fn,
                mesh=mesh,
                in_specs=(None, P(None, "mp")),
                out_specs=P(None),
                check_rep=False,
            )(x, kernel)
        return output


def make_causal_mask(q_len, k_len):
    i = jnp.arange(q_len)[:, None]
    j = jnp.arange(k_len)[None, :]
    return jnp.where(i >= j - (k_len - q_len), 0.0, -1e9)


class FullyParallelQwenAttention(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    rope_mode: str = "cos_sin"  # "cos_sin" or "complex"
    cache_mode: str = "pkv"      # "pkv" or "mutable"

    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        self.num_heads = c["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = c.get("num_key_value_heads", self.num_heads)
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.rope_theta = c.get("rope_theta", 1000000.0)

        self.q_proj = ParallelDense(self.hidden_size, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, use_bias=True, name="q_proj")
        self.k_proj = ParallelDense(self.kv_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, use_bias=True, name="k_proj")
        self.v_proj = ParallelDense(self.kv_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, use_bias=True, name="v_proj")
        self.o_proj = ParallelDense(self.hidden_size, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, use_bias=False, name="o_proj")

        if self.rope_mode == "complex":
            self.freqs_cis = precompute_freqs_cis(self.head_dim, c.get("max_position_embeddings", 2048), self.rope_theta)

        if self.cache_mode == "mutable":
            max_seq_len = c.get("max_position_embeddings", 2048)
            self.cached_key = self.variable("cache", "cached_key", jnp.zeros, (1, max_seq_len, self.num_kv_heads, self.head_dim), jnp.bfloat16)
            self.cached_value = self.variable("cache", "cached_value", jnp.zeros, (1, max_seq_len, self.num_kv_heads, self.head_dim), jnp.bfloat16)
            self.cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        batch, seq, _ = hidden_states.shape
        q = self.q_proj(hidden_states).reshape(batch, seq, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(batch, seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(batch, seq, self.num_kv_heads, self.head_dim)

        if position_ids is not None:
            if self.rope_mode == "complex":
                freqs_cis = self.freqs_cis[position_ids]
                q, k = apply_rotary_emb_complex(q, k, freqs_cis)
            else:
                cos, sin = compute_cos_sin_cache(position_ids, self.head_dim, self.rope_theta)
                q, k = apply_rotary_emb(q, k, cos, sin)

        if self.cache_mode == "mutable":
            cur_index = self.cache_index.value
            if past_key_value is not None:
                past_k, past_v = past_key_value
            else:
                past_k = self.cached_key.value[:, :cur_index, :, :]
                past_v = self.cached_value.value[:, :cur_index, :, :]
            k_full = jnp.concatenate([past_k, k], axis=1)
            v_full = jnp.concatenate([past_v, v], axis=1)
            self.cached_key.value = self.cached_key.value.at[:, cur_index:cur_index + seq, :, :].set(k.astype(jnp.bfloat16))
            self.cached_value.value = self.cached_value.value.at[:, cur_index:cur_index + seq, :, :].set(v.astype(jnp.bfloat16))
            self.cache_index.value = cur_index + seq
            cache_k, cache_v = k_full, v_full
        else:
            if past_key_value is not None:
                past_k, past_v = past_key_value
                k = jnp.concatenate([past_k, k], axis=1)
                v = jnp.concatenate([past_v, v], axis=1)
            cache_k, cache_v = k, v

        k_for_attn = cache_k
        v_for_attn = cache_v

        if self.num_heads != self.num_kv_heads:
            repeat = self.num_heads // self.num_kv_heads
            k_for_attn = jnp.repeat(k_for_attn, repeat, axis=2)
            v_for_attn = jnp.repeat(v_for_attn, repeat, axis=2)

        q = q.transpose(0, 2, 1, 3)
        k_t = k_for_attn.transpose(0, 2, 1, 3)
        v_t = v_for_attn.transpose(0, 2, 1, 3)

        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.einsum('bhqd,bhkd->bhqk', q, k_t) * scale
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
        attn_out = jnp.einsum('bhqk,bhkd->bhqd', probs, v_t)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq, self.hidden_size)
        attn_out = self.o_proj(attn_out)
        return attn_out, (cache_k, cache_v)


class QwenMLP(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        self.intermediate_size = c.get("intermediate_size", 4 * c["hidden_size"])
        self.gate_proj = ParallelDense(self.intermediate_size, dtype=self.dtype, param_dtype=self.dtype, name="gate_proj")
        self.up_proj = ParallelDense(self.intermediate_size, dtype=self.dtype, param_dtype=self.dtype, name="up_proj")
        self.down_proj = ParallelDense(c["hidden_size"], dtype=self.dtype, param_dtype=self.dtype, name="down_proj")

    def __call__(self, x):
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenDecoderLayer(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    rope_mode: str = "cos_sin"
    cache_mode: str = "pkv"

    def setup(self):
        c = self.config
        self.input_layernorm = nn.RMSNorm(epsilon=c.get("rms_norm_eps", 1e-6), dtype=jnp.bfloat16, name="input_layernorm")
        self.self_attn = FullyParallelQwenAttention(config=c, dtype=jnp.bfloat16, rope_mode=self.rope_mode, cache_mode=self.cache_mode)
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
    rope_mode: str = "cos_sin"
    lm_head_kind: str = "parallel"  # "parallel" or "dense"
    cache_mode: str = "pkv"          # "pkv" or "mutable"
    embed_kind: str = "parallel"     # "parallel" or "standard"

    def setup(self):
        c = self.config
        if self.embed_kind == "standard":
            self.embed_tokens = StandardEmbed(c["vocab_size"], c["hidden_size"], dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name="embed_tokens")
        else:
            self.embed_tokens = ParallelEmbed(c["vocab_size"], c["hidden_size"], dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name="embed_tokens")
        self.layers = [QwenDecoderLayer(config=c, dtype=jnp.bfloat16, rope_mode=self.rope_mode, cache_mode=self.cache_mode, name=f"layers_{i}") for i in range(c["num_hidden_layers"])]
        self.norm = nn.RMSNorm(epsilon=c.get("rms_norm_eps", 1e-6), dtype=jnp.bfloat16, name="norm")
        if self.lm_head_kind == "parallel":
            self.lm_head = ParallelDense(c["vocab_size"], dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name="lm_head")
        else:
            self.lm_head = nn.Dense(c["vocab_size"], dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, use_bias=False, name="lm_head")

    def __call__(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, return_dict=True):
        batch, seq = input_ids.shape
        key_len = seq if past_key_values is None or past_key_values[0] is None else past_key_values[0][0].shape[1] + seq
        if attention_mask is None:
            attention_mask = jnp.ones((batch, 1, seq, key_len), dtype=self.dtype)
        causal_mask = make_causal_mask(seq, key_len)[None, None, :, :]
        attention_bias = jnp.where(attention_mask == 0, -1e9, 0.0) + causal_mask

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


# -------- Weight loading ---------
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
    print(f"Loading JAX model weights from {model_path}...")
    params = {"params": {}}
    loaded_count = 0
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
                        loaded_count += 1
    gc.collect()
    print(f"Weight loading completed. Loaded {loaded_count} parameters.")
    return params


# -------- Generation ---------
def sample_next_token(logits):
    return int(jnp.argmax(logits, axis=-1)[0])


def generate_text(model, params, tokenizer, max_tokens, prompt):
    print("Starting text generation...")
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024**3
    print(f"Initial memory before generation: {initial_memory:.2f} GB used")
    print(f"Free memory: {psutil.virtual_memory().available / 1024**3:.2f} GB")

    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(formatted_text, return_tensors="jax")
    batch, seq = input_ids.shape
    position_ids = jnp.arange(seq, dtype=jnp.int32)[None, :]
    past_key_values = None

    generated_tokens = []
    start_time = time.time()
    peak_memory = initial_memory

    print(f"Entering generation loop for {max_tokens} tokens...")
    for i in range(max_tokens):
        current_seq_len = input_ids.shape[1]
        key_len = current_seq_len if past_key_values is None or past_key_values[0] is None else past_key_values[0][0].shape[1] + current_seq_len
        attention_mask = jnp.ones((batch, 1, current_seq_len, key_len), dtype=jnp.float32)

        # If model uses mutable cache, we must pass mutable collections to apply
        if isinstance(model, Qwen25ForCausalLM) and model.cache_mode == "mutable":
            outputs, mutable_vars = model.apply(params, input_ids=input_ids, attention_mask=attention_mask,
                                               position_ids=position_ids, past_key_values=past_key_values, return_dict=True,
                                               mutable=["cache"])  # ensure cache is updated
        else:
            outputs = model.apply(params, input_ids=input_ids, attention_mask=attention_mask,
                                  position_ids=position_ids, past_key_values=past_key_values, return_dict=True)

        logits = outputs["logits"]
        past_key_values = outputs["past_key_values"]

        next_token = sample_next_token(logits[:, -1, :])
        generated_tokens.append(int(next_token))
        input_ids = jnp.array([[next_token]])
        position_ids = position_ids[:, -1:] + 1

        current_mem = psutil.virtual_memory().used / (1024**3)
        if current_mem > peak_memory:
            peak_memory = current_mem

        token_text = tokenizer.decode(int(next_token), skip_special_tokens=True)
        print(f"{i+1:03d}: '{token_text}'")
        if int(next_token) == tokenizer.eos_token_id or "<|im_end|>" in token_text:
            print("Stopping generation: EOS token encountered.")
            break

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_token = total_time / max(1, len(generated_tokens))
    print(f"Memory after generation: {psutil.virtual_memory().used / (1024**3):.2f} GB used")
    print(f"Peak memory during generation: {peak_memory:.2f} GB used")
    print(f"Free memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print(f"Total tokens generated: {len(generated_tokens)}")
    print(f"Average time per token: {avg_time_per_token:.2f} seconds")

    full_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print("Generation complete.")
    return full_output, peak_memory, avg_time_per_token


# -------- Device mesh ---------
def setup_device_mesh():
    global mesh
    print("Setting up device mesh...")
    devices = jax.devices()
    print(f"Available devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device}")
    mesh = Mesh(devices, axis_names=("mp",))
    print(f"Created mesh: {mesh}")
    return mesh


# -------- Main ---------
def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B-Instruct JAX Inference (gpt5 variant)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--rope", type=str, default="cos_sin", choices=["cos_sin", "complex"], help="RoPE implementation")
    parser.add_argument("--lm_head", type=str, default="parallel", choices=["parallel", "dense"], help="LM head implementation")
    parser.add_argument("--cache_mode", type=str, default="pkv", choices=["pkv", "mutable"], help="KV cache strategy")
    parser.add_argument("--embed", type=str, default="parallel", choices=["parallel", "standard"], help="Embedding implementation")
    parser.add_argument("--tokens", type=int, default=500, help="Max generation tokens")
    parser.add_argument("--prompt", type=str, default="Question: Sam scores 80 on the first test and 90 on the second. What score does he need on the third test to have an average of 85?", help="Prompt to generate for")
    args = parser.parse_args()

    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    setup_device_mesh()

    with open(os.path.join(args.model_path, "config.json")) as f:
        config = json.load(f)

    model = Qwen25ForCausalLM(config=config, dtype=dtype, rope_mode=args.rope, lm_head_kind=args.lm_head,
                               cache_mode=args.cache_mode, embed_kind=args.embed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    params = load_params(model, args.model_path, dtype)

    print("\n" + "="*80)
    print("Prompt:")
    print(args.prompt)
    output, peak_mem, avg_time_per_token = generate_text(model, params, tokenizer, args.tokens, args.prompt)
    print(f"Output: {output}")
    print(f"Peak memory: {peak_mem:.2f} GB")
    print(f"Avg time per token: {avg_time_per_token:.4f} seconds")
    print("="*80)


if __name__ == "__main__":
    main()