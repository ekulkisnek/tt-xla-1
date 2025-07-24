#!/usr/bin/env python3
"""
Systematic debugging script following the battle plan to identify exact divergence points.
Uses Qwen2.5-0.5B-Instruct for fast iteration.
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import flax
import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
import json

from q25_debug import Qwen25ForCausalLM, load_params, apply_chat_template

def step0_load_models():
    """Step 0: Load both models using 0.5B for fast iteration"""
    model_path = "weights_05b"
    
    print("="*60)
    print("STEP 0: LOADING 0.5B MODELS")
    print("="*60)
    
    # Load PyTorch model
    print("Loading PyTorch model...")
    pt_model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    
    # Load config and JAX model
    print("Loading JAX model...")
    with open(os.path.join(model_path, "config.json"), 'r') as f:
        config = json.load(f)
    
    jax_model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
    jax_params = load_params(jax_model, model_path, jnp.float32)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"✅ Models loaded successfully")
    print(f"Config: {config['num_hidden_layers']} layers, {config['hidden_size']} hidden")
    
    return pt_model, jax_model, jax_params, tokenizer, config

def step1_compare_state_dicts(pt_model, jax_model, jax_params):
    """Step 1: Load both state-dicts into Python, side by side"""
    print("\n" + "="*60)
    print("STEP 1: COMPARING STATE DICTS")
    print("="*60)
    
    # Get PyTorch state dict
    PT = pt_model.state_dict()
    
    # Flatten JAX params
    FX = flax.traverse_util.flatten_dict(jax_params, sep='.')
    
    # Create weight mapping
    def pytorch_to_jax_key(k_torch):
        """Map PyTorch keys to JAX keys"""
        if k_torch == "model.embed_tokens.weight":
            return "params.embed_tokens.embedding"
        elif k_torch == "model.norm.weight":
            return "params.norm.scale"
        elif k_torch == "lm_head.weight":
            return "params.lm_head.kernel"
        elif "model.layers." in k_torch:
            # Parse layer number
            parts = k_torch.split(".")
            layer_num = parts[2]
            rest = ".".join(parts[3:])
            
            if rest == "input_layernorm.weight":
                return f"params.layers_{layer_num}.input_layernorm.scale"
            elif rest == "post_attention_layernorm.weight":
                return f"params.layers_{layer_num}.post_attention_layernorm.scale"
            elif rest.startswith("self_attn."):
                proj = rest.split(".")[1]  # q_proj, k_proj, etc.
                if rest.endswith(".weight"):
                    return f"params.layers_{layer_num}.self_attn.{proj}.kernel"
                elif rest.endswith(".bias"):
                    return f"params.layers_{layer_num}.self_attn.{proj}.bias"
            elif rest.startswith("mlp."):
                proj = rest.split(".")[1]  # gate_proj, up_proj, down_proj
                if rest.endswith(".weight"):
                    return f"params.layers_{layer_num}.mlp.{proj}.kernel"
        return None
    
    print("Scanning for shape mismatches...")
    issues = []
    
    for k_torch, v_torch in PT.items():
        if 'rotary' in k_torch:  # skip rope caches
            continue
            
        k_jax = pytorch_to_jax_key(k_torch)
        if k_jax is None:
            print(f"⚠️  NO MAPPING: {k_torch}")
            continue
            
        if k_jax not in FX:
            print(f"❌ MISSING: {k_torch} -> {k_jax}")
            issues.append(f"missing_{k_torch}")
        else:
            v_jax = FX[k_jax]
            if v_torch.shape != v_jax.shape:
                if v_torch.shape == v_jax.shape[::-1]:  # Transposed
                    print(f"🔄 LIKELY TRANSPOSE ERROR: {k_torch}")
                    print(f"   PyTorch: {v_torch.shape} vs JAX: {v_jax.shape}")
                    issues.append(f"transpose_{k_torch}")
                else:
                    print(f"❌ SHAPE MISMATCH: {k_torch}")
                    print(f"   PyTorch: {v_torch.shape} vs JAX: {v_jax.shape}")
                    issues.append(f"shape_{k_torch}")
            else:
                # Check if values are close
                diff = np.abs(v_torch.detach().numpy() - np.array(v_jax)).max()
                if diff > 1e-4:
                    print(f"⚠️  VALUE DIFF: {k_torch}, max_diff={diff:.6f}")
                    issues.append(f"value_{k_torch}")
    
    print(f"\n📊 Found {len(issues)} issues")
    return issues

def step2_first_token_sanity_check(pt_model, jax_model, jax_params, tokenizer):
    """Step 2: First-token logits sanity check"""
    print("\n" + "="*60)
    print("STEP 2: FIRST-TOKEN LOGITS SANITY CHECK")
    print("="*60)
    
    prompt = "2 + 3 ="
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, 
        add_generation_prompt=True
    )
    ids = tokenizer(text, return_tensors="pt").input_ids
    
    print(f"Prompt: {repr(prompt)}")
    print(f"Template: {repr(text)}")
    print(f"Input IDs: {ids.tolist()}")
    
    # PyTorch forward pass
    with torch.no_grad():
        pt_outputs = pt_model(ids)
        pt_logits = pt_outputs.logits[0, -1].detach().numpy()
    
    # JAX forward pass
    jax_outputs = jax_model.apply(jax_params, jnp.array(ids.numpy()))
    jax_logits = np.array(jax_outputs['logits'][0, -1])
    
    # Compare argmax
    pt_argmax = pt_logits.argmax()
    jax_argmax = jax_logits.argmax()
    
    pt_token = tokenizer.decode(pt_argmax)
    jax_token = tokenizer.decode(int(jax_argmax))
    
    print(f"PyTorch argmax: {pt_argmax} -> {repr(pt_token)}")
    print(f"JAX argmax: {jax_argmax} -> {repr(jax_token)}")
    
    # Check top-5 for both
    pt_top5 = np.argsort(pt_logits)[-5:][::-1]
    jax_top5 = np.argsort(jax_logits)[-5:][::-1]
    
    print("\nPyTorch top-5:")
    for i, idx in enumerate(pt_top5):
        token = tokenizer.decode(idx)
        print(f"  {i+1}. {idx}: {pt_logits[idx]:.4f} -> {repr(token)}")
    
    print("\nJAX top-5:")
    for i, idx in enumerate(jax_top5):
        token = tokenizer.decode(idx)
        print(f"  {i+1}. {idx}: {jax_logits[idx]:.4f} -> {repr(token)}")
    
    # Overall difference
    max_diff = np.abs(pt_logits - jax_logits).max()
    print(f"\nMax logit difference: {max_diff:.6f}")
    
    tokens_match = (pt_token == jax_token)
    print(f"✅ Tokens match: {tokens_match}" if tokens_match else f"❌ Tokens differ: {pt_token} vs {jax_token}")
    
    return tokens_match, max_diff

def step3_layer_by_layer_probe(pt_model, jax_model, jax_params, tokenizer):
    """Step 3: Layer-by-layer probe to find exact divergence point"""
    print("\n" + "="*60)
    print("STEP 3: LAYER-BY-LAYER PROBE")
    print("="*60)
    
    # Use simple input for layer comparison
    prompt = "Hello"
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, 
        add_generation_prompt=True
    )
    ids = tokenizer(text, return_tensors="pt").input_ids
    
    print(f"Probe input: {repr(text)}")
    print(f"Input shape: {ids.shape}")
    
    def dump_layer_outputs_pt(model, ids):
        """Extract PyTorch layer outputs"""
        h = model.model.embed_tokens(ids)
        outs = [h.detach().numpy()]  # Start with embeddings
        
        for i, layer in enumerate(model.model.layers):
            # Input norm
            h_norm = layer.input_layernorm(h)
            # Self attention
            attn_out = layer.self_attn(h_norm)[0]
            h = h + attn_out
            # Post attention norm  
            h_norm2 = layer.post_attention_layernorm(h)
            # MLP
            mlp_out = layer.mlp(h_norm2)
            h = h + mlp_out
            outs.append(h.detach().numpy())
        
        return outs
    
    def dump_layer_outputs_jax(model, params, ids):
        """Extract JAX layer outputs"""
        # Embeddings
        h = model.embed_tokens.apply({"params": params["params"]["embed_tokens"]}, ids)
        outs = [np.array(h)]
        
        # Layers
        past_key_values = [None] * model.num_layers
        for i, layer in enumerate(model.layers):
            layer_params = {"params": params["params"][f"layers_{i}"]}
            h, _ = layer.apply(
                layer_params, 
                h, 
                attention_mask=None, 
                position_ids=None,
                past_key_value=past_key_values[i]
            )
            outs.append(np.array(h))
        
        return outs
    
    print("Computing PyTorch layer outputs...")
    with torch.no_grad():
        pt_outs = dump_layer_outputs_pt(pt_model, ids)
    
    print("Computing JAX layer outputs...")
    jax_outs = dump_layer_outputs_jax(jax_model, jax_params, jnp.array(ids.numpy()))
    
    print(f"\nComparing {len(pt_outs)} layer outputs...")
    
    first_divergence = None
    for i, (pt_out, jax_out) in enumerate(zip(pt_outs, jax_outs)):
        diff = np.abs(pt_out - jax_out).max()
        
        if i == 0:
            print(f"Layer {i:2d} (embeddings): max_diff = {diff:.6f}")
        else:
            print(f"Layer {i:2d} (layer_{i-1}):  max_diff = {diff:.6f}")
        
        if diff > 1e-4 and first_divergence is None:
            first_divergence = i
            print(f"🚨 FIRST DIVERGENCE at layer {i}!")
            break
    
    return first_divergence

def main():
    """Run the systematic debugging process"""
    print("🔍 SYSTEMATIC PYTORCH vs JAX DEBUGGING")
    print("Using Qwen2.5-0.5B-Instruct for fast iteration")
    
    # Step 0: Load models
    pt_model, jax_model, jax_params, tokenizer, config = step0_load_models()
    
    # Step 1: Compare state dicts
    issues = step1_compare_state_dicts(pt_model, jax_model, jax_params)
    
    # Step 2: First token check
    tokens_match, max_diff = step2_first_token_sanity_check(pt_model, jax_model, jax_params, tokenizer)
    
    if not tokens_match:
        print(f"\n⚠️  Tokens don't match, investigating layer-by-layer...")
        # Step 3: Layer-by-layer probe
        divergence_layer = step3_layer_by_layer_probe(pt_model, jax_model, jax_params, tokenizer)
        
        if divergence_layer is not None:
            print(f"\n🎯 DIVERGENCE FOUND AT LAYER {divergence_layer}")
            print(f"Focus your debugging on this layer!")
        else:
            print(f"\n🤔 No clear divergence found, but max_diff = {max_diff:.6f}")
    else:
        print(f"\n🎉 SUCCESS! Tokens match with max_diff = {max_diff:.6f}")
        print(f"You can now test with the 7B model!")
    
    print(f"\n📋 SUMMARY:")
    print(f"  Issues found: {len(issues)}")
    print(f"  Tokens match: {tokens_match}")
    print(f"  Max logit diff: {max_diff:.6f}")

if __name__ == "__main__":
    main() 