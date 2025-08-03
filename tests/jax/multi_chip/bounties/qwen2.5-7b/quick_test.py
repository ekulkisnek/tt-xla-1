#!/usr/bin/env python3
"""
Quick test script to verify each optimization works and measure time per token
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

def test_optimization(optimization_name, model_class):
    """Test a specific optimization."""
    print(f"\n{'='*60}")
    print(f"Testing {optimization_name}")
    print(f"{'='*60}")
    
    # Load model and tokenizer
    model_path = "weights"
    with open(os.path.join(model_path, "config.json")) as f:
        config = json.load(f)
    
    model = model_class(config=config, dtype=jnp.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    params = load_params(model, model_path, jnp.bfloat16)
    
    # Create JIT function
    jit_apply = jax.jit(model.apply, static_argnames=['return_dict'])
    
    # Simple test prompt
    test_prompt = "The answer is"
    inputs = tokenizer(test_prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    position_ids = jax.numpy.arange(input_ids.shape[1])[None, :]
    
    print(f"Input prompt: '{test_prompt}'")
    print(f"Input tokens: {input_ids[0].tolist()}")
    
    # Generate a few tokens
    past_key_values = None
    generated_tokens = input_ids[0].tolist()
    times = []
    
    print("\nGenerating tokens:")
    for i in range(20):  # Generate 20 tokens to see performance benefits
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
    
    avg_time = sum(times) / len(times)
    print(f"\nResults for {optimization_name}:")
    print(f"  Average time per token: {avg_time:.4f} seconds")
    print(f"  Tokens per second: {1/avg_time:.2f}")
    print(f"  Final text: '{full_text}'")
    
    return avg_time, full_text

def main():
    print("Quick test of optimizations - generating 20 tokens each")
    
    # Import the different model versions
    import importlib.util
    
    # Test original - copy the file to have .py extension
    import shutil
    shutil.copy("q25j7fast1", "q25j7fast1.py")
    spec1 = importlib.util.spec_from_file_location("q25j7fast1", "q25j7fast1.py")
    q25j7fast1 = importlib.util.module_from_spec(spec1)
    spec1.loader.exec_module(q25j7fast1)
    
    # Test optimization 1
    spec2 = importlib.util.spec_from_file_location("q25j7fast_opt1", "q25j7fast_opt1.py")
    q25j7fast_opt1 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(q25j7fast_opt1)
    
    # Test optimization 2
    spec3 = importlib.util.spec_from_file_location("q25j7fast_opt2", "q25j7fast_opt2.py")
    q25j7fast_opt2 = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(q25j7fast_opt2)
    
    # Test optimization 3
    spec4 = importlib.util.spec_from_file_location("q25j7fast_opt3", "q25j7fast_opt3.py")
    q25j7fast_opt3 = importlib.util.module_from_spec(spec4)
    spec4.loader.exec_module(q25j7fast_opt3)
    
    # Test optimization 4 (ALL optimizations combined)
    spec5 = importlib.util.spec_from_file_location("q25j7fast_opt4", "q25j7fast_opt4.py")
    q25j7fast_opt4 = importlib.util.module_from_spec(spec5)
    spec5.loader.exec_module(q25j7fast_opt4)
    
    results = {}
    
    # Test each version
    try:
        results["Original"] = test_optimization("Original", q25j7fast1.Qwen25ForCausalLM)
    except Exception as e:
        print(f"Original failed: {e}")
        results["Original"] = (float('inf'), "FAILED")
    
    try:
        results["Opt1 (Precomputed RoPE)"] = test_optimization("Opt1 (Precomputed RoPE)", q25j7fast_opt1.Qwen25ForCausalLM)
    except Exception as e:
        print(f"Opt1 failed: {e}")
        results["Opt1 (Precomputed RoPE)"] = (float('inf'), "FAILED")
    
    try:
        results["Opt2 (JIT)"] = test_optimization("Opt2 (JIT)", q25j7fast_opt2.Qwen25ForCausalLM)
    except Exception as e:
        print(f"Opt2 failed: {e}")
        results["Opt2 (JIT)"] = (float('inf'), "FAILED")
    
    try:
        results["Opt3 (Precomputed Causal Mask)"] = test_optimization("Opt3 (Precomputed Causal Mask)", q25j7fast_opt3.Qwen25ForCausalLM)
    except Exception as e:
        print(f"Opt3 failed: {e}")
        results["Opt3 (Precomputed Causal Mask)"] = (float('inf'), "FAILED")
    
    try:
        results["Opt4 (ALL Optimizations)"] = test_optimization("Opt4 (ALL Optimizations)", q25j7fast_opt4.Qwen25ForCausalLM)
    except Exception as e:
        print(f"Opt4 failed: {e}")
        results["Opt4 (ALL Optimizations)"] = (float('inf'), "FAILED")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    original_time = results["Original"][0]
    for name, (time_per_token, text) in results.items():
        if time_per_token != float('inf'):
            speedup = original_time / time_per_token
            print(f"{name}:")
            print(f"  Time per token: {time_per_token:.4f}s")
            print(f"  Speedup vs original: {speedup:.2f}x")
            print(f"  Text: '{text}'")
        else:
            print(f"{name}: FAILED")
        print()

if __name__ == "__main__":
    main() 