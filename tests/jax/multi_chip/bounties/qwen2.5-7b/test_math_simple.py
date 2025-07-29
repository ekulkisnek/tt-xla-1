#!/usr/bin/env python3
"""
Simple Mathematical Reasoning Test
Quick test to compare JAX vs PyTorch mathematical reasoning quality.
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
    # Look for patterns like "The answer is X", "= X", or just numbers
    patterns = [
        r"(?:answer\s+is|answer:\s*|=\s*)(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*$"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response.lower())
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                continue
    
    # Find any number in the response
    numbers = re.findall(r'\d+(?:\.\d+)?', response)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    
    return None

def test_math_problem(jax_model, jax_params, torch_model, tokenizer, problem, expected):
    """Test a single math problem"""
    print(f"\n🧮 Testing: {problem}")
    print(f"   Expected: {expected}")
    
    # Format prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant that solves math problems."},
        {"role": "user", "content": f"Solve this math problem step by step: {problem}"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    try:
        # JAX generation
        print("   Generating JAX response...")
        jax_response = generate_text(
            model=jax_model,
            params=jax_params,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=256,
            temperature=0.1,
            use_chat_template=False,
            use_enhanced_sampling=True
        )
        
        # PyTorch generation
        print("   Generating PyTorch response...")
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = torch_model.generate(
                inputs.input_ids,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        torch_response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # Extract answers
        jax_answer = extract_number(jax_response)
        torch_answer = extract_number(torch_response)
        
        # Check correctness
        jax_correct = jax_answer is not None and abs(jax_answer - expected) < 0.01
        torch_correct = torch_answer is not None and abs(torch_answer - expected) < 0.01
        
        print(f"   JAX Answer: {jax_answer} {'✅' if jax_correct else '❌'}")
        print(f"   PyTorch Answer: {torch_answer} {'✅' if torch_correct else '❌'}")
        
        # Show short responses
        print(f"   JAX Response: {jax_response[:100]}...")
        print(f"   PyTorch Response: {torch_response[:100]}...")
        
        return {
            'problem': problem,
            'expected': expected,
            'jax_answer': jax_answer,
            'torch_answer': torch_answer,
            'jax_correct': jax_correct,
            'torch_correct': torch_correct,
            'jax_response': jax_response,
            'torch_response': torch_response
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {
            'problem': problem,
            'expected': expected,
            'error': str(e),
            'jax_correct': False,
            'torch_correct': False
        }

def main():
    """Run simple mathematical reasoning tests"""
    print("🚀 Simple Mathematical Reasoning Test")
    print("="*60)
    
    # Load config and tokenizer
    config_path = "./weights/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained("./weights")
    
    # Load JAX model
    print("\n📥 Loading JAX model...")
    jax_model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    jax_params = load_params(jax_model, "./weights", jnp.bfloat16)
    
    # Load PyTorch model
    print("📥 Loading PyTorch model...")
    torch_model = AutoModelForCausalLM.from_pretrained(
        "./weights",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    torch_model.eval()
    
    # Test problems
    test_problems = [
        ("What is 23 + 47?", 70),
        ("Calculate 156 - 89.", 67),
        ("What is 12 × 15?", 180),
        ("Sarah has 24 apples. She gives 8 apples away and buys 15 more. How many apples does she have?", 31),
        ("What is 25% of 80?", 20)
    ]
    
    print(f"\n🧮 Testing {len(test_problems)} mathematical problems...")
    
    results = []
    jax_correct = 0
    torch_correct = 0
    
    for problem, expected in test_problems:
        result = test_math_problem(jax_model, jax_params, torch_model, tokenizer, problem, expected)
        results.append(result)
        
        if result.get('jax_correct', False):
            jax_correct += 1
        if result.get('torch_correct', False):
            torch_correct += 1
    
    # Summary
    print("\n" + "="*60)
    print("📊 MATHEMATICAL REASONING TEST RESULTS")
    print("="*60)
    
    total = len(test_problems)
    jax_accuracy = jax_correct / total
    torch_accuracy = torch_correct / total
    
    print(f"📊 Results:")
    print(f"   • Total problems: {total}")
    print(f"   • JAX accuracy: {jax_correct}/{total} ({jax_accuracy:.1%})")
    print(f"   • PyTorch accuracy: {torch_correct}/{total} ({torch_accuracy:.1%})")
    
    # Quality assessment
    if jax_accuracy >= 0.8:
        print(f"✅ JAX mathematical reasoning quality is GOOD ({jax_accuracy:.1%})")
    elif jax_accuracy >= 0.6:
        print(f"⚠️ JAX mathematical reasoning quality is ACCEPTABLE ({jax_accuracy:.1%})")
    else:
        print(f"❌ JAX mathematical reasoning quality needs IMPROVEMENT ({jax_accuracy:.1%})")
    
    print(f"\n🎯 JAX vs PyTorch comparison:")
    if jax_accuracy >= torch_accuracy:
        print(f"✅ JAX is as good as or better than PyTorch for mathematical reasoning")
    else:
        print(f"⚠️ PyTorch outperforms JAX in mathematical reasoning")
    
    print("="*60)
    
    # Clean up
    del jax_model, jax_params, torch_model
    jax.clear_caches()

if __name__ == "__main__":
    main() 