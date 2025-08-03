#!/usr/bin/env python3
"""
Test only Optimization 7: Efficient Attention Bias with lax.select
"""
import os
import sys
import json
import time
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
from safetensors import safe_open

# Disable x64 globally for faster inference
os.environ["JAX_ENABLE_X64"] = "0"

# XLA flags for CPU optimization - enable multi-threading
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true"
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

import importlib.util

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
    print("Weight loading completed")
    return params

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

def sample_next_token(logits):
    """Simplified greedy sampling only."""
    return int(jnp.argmax(logits, axis=-1)[0])

def test_optimization7():
    """Test Optimization 7: Efficient Attention Bias with lax.select."""
    print("Testing Optimization 7: Efficient Attention Bias with lax.select")
    print("=" * 70)
    
    # Load the optimized model
    spec = importlib.util.spec_from_file_location("q25j7fast_opt7", "q25j7fast_opt7.py")
    q25j7fast_opt7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q25j7fast_opt7)
    
    # Load model and tokenizer
    model_path = "weights"
    with open(os.path.join(model_path, "config.json")) as f:
        config = json.load(f)
    
    model = q25j7fast_opt7.Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    params = load_params(model, model_path, jnp.bfloat16)
    
    # Create JIT function
    jit_apply = jax.jit(model.apply, static_argnames=['return_dict'])
    
    # Test prompt
    test_prompt = "Please solve this math problem step by step: If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?"
    inputs = tokenizer(test_prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    position_ids = jax.numpy.arange(input_ids.shape[1])[None, :]
    
    print(f"Input prompt: '{test_prompt}'")
    print(f"Input tokens: {input_ids[0].tolist()}")
    print(f"Input length: {len(input_ids[0])} tokens")
    
    # Generate tokens
    past_key_values = None
    generated_tokens = input_ids[0].tolist()
    times = []
    
    print("\nGenerating tokens:")
    for i in range(15):  # Generate 15 tokens
        start_time = time.perf_counter()
        outputs = jit_apply(params, input_ids=input_ids, position_ids=position_ids, past_key_values=past_key_values, return_dict=True)
        logits = outputs["logits"]
        past_key_values = outputs["past_key_values"]
        next_token = sample_next_token(logits[:, -1, :])
        token_time = time.perf_counter() - start_time
        times.append(token_time)
        
        generated_tokens.append(int(next_token))
        input_ids = jax.numpy.array([[next_token]])
        position_ids = position_ids[:, -1:] + 1
        
        # Decode and show the token
        token_text = tokenizer.decode(int(next_token), skip_special_tokens=True)
        full_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"  Token {i+1}: '{token_text}' (id: {next_token}) - Time: {token_time:.4f}s")
        print(f"  Full text so far: '{full_text}'")
        
        if next_token == tokenizer.eos_token_id:
            break
    
    avg_time = sum(times) / len(times)
    print(f"\nResults for Optimization 7:")
    print(f"  Average time per token: {avg_time:.4f} seconds")
    print(f"  Tokens per second: {1/avg_time:.2f}")
    print(f"  Final text: '{full_text}'")
    
    # Compare with baseline from results file
    baseline_time = 9.7783  # From optimization_results.md
    speedup = baseline_time / avg_time
    print(f"  Speedup vs original: {speedup:.2f}x")
    
    return avg_time, full_text, speedup

if __name__ == "__main__":
    test_optimization7() 