#!/usr/bin/env python3
"""
Systematic Mathematical Reasoning Test
Disable broken enhanced generation and test standard generation systematically.
"""

import os
import json
import re
import numpy as np
import torch
import jax
import jax.numpy as jnp
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import our JAX implementation
from qwen_jax_inference import Qwen25ForCausalLM, load_params, generate_text

def extract_number(response: str) -> float:
    """Extract the final number from a response"""
    numbers = re.findall(r'\d+(?:\.\d+)?', response)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    return None

def test_single_math_problem_systematic():
    """Test one math problem systematically with different generation modes"""
    print("🔬 Systematic Mathematical Reasoning Test")
    print("="*60)
    
    # Load models
    config_path = "./weights/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained("./weights")
    
    print("📥 Loading JAX model...")
    jax_model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    jax_params = load_params(jax_model, "./weights", jnp.bfloat16)
    
    print("📥 Loading PyTorch model...")
    torch_model = AutoModelForCausalLM.from_pretrained(
        "./weights",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    torch_model.eval()
    
    # Simple test problem
    problem = "What is 23 + 47?"
    expected = 70
    
    print(f"\n🧮 Testing: {problem}")
    print(f"Expected: {expected}")
    
    # Format prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant that solves math problems."},
        {"role": "user", "content": f"Solve this math problem: {problem}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print(f"\n📝 Formatted prompt: {repr(prompt[:100])}...")
    
    # Test 1: JAX with STANDARD generation (no enhanced features)
    print(f"\n🔍 Test 1: JAX Standard Generation")
    try:
        jax_response_standard = generate_text(
            model=jax_model,
            params=jax_params,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=100,
            temperature=0.1,
            use_chat_template=False,
            use_enhanced_sampling=False,    # DISABLE enhanced sampling
            use_enhanced_generation=False   # DISABLE enhanced generation
        )
        
        jax_answer_standard = extract_number(jax_response_standard)
        jax_correct_standard = jax_answer_standard is not None and abs(jax_answer_standard - expected) < 0.01
        
        print(f"   Response: {jax_response_standard[:100]}...")
        print(f"   Answer: {jax_answer_standard} {'✅' if jax_correct_standard else '❌'}")
        
    except Exception as e:
        print(f"   ❌ Standard generation failed: {e}")
        jax_correct_standard = False
        jax_response_standard = "ERROR"
    
    # Test 2: PyTorch reference
    print(f"\n🔍 Test 2: PyTorch Reference")
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = torch_model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    torch_response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    
    torch_answer = extract_number(torch_response)
    torch_correct = torch_answer is not None and abs(torch_answer - expected) < 0.01
    
    print(f"   Response: {torch_response[:100]}...")
    print(f"   Answer: {torch_answer} {'✅' if torch_correct else '❌'}")
    
    # Analysis
    print(f"\n📊 Systematic Analysis:")
    print(f"   • JAX Standard Generation: {'✅ WORKS' if jax_correct_standard else '❌ BROKEN'}")
    print(f"   • PyTorch Reference: {'✅ WORKS' if torch_correct else '❌ BROKEN'}")
    
    if jax_correct_standard and torch_correct:
        print(f"✅ ROOT CAUSE IDENTIFIED: Enhanced generation loop is broken, standard generation works")
        print(f"✅ SOLUTION: Fix or disable enhanced generation features")
    elif jax_correct_standard and not torch_correct:
        print(f"⚠️ UNEXPECTED: JAX works but PyTorch doesn't - check PyTorch setup")
    elif not jax_correct_standard and torch_correct:
        print(f"❌ DEEPER ISSUE: Even standard JAX generation is broken - fundamental model issue")
        print(f"❌ NEXT STEP: Debug basic JAX model implementation")
    else:
        print(f"❌ BOTH BROKEN: Check model loading and tokenizer setup")
    
    # Clean up
    del jax_model, jax_params, torch_model
    jax.clear_caches()

if __name__ == "__main__":
    test_single_math_problem_systematic() 