#!/usr/bin/env python3
"""
Simple test script for PyTorch Qwen2.5-7B with dog food prompt only
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import gc
import psutil
import os

def check_memory():
    """Check available system memory"""
    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    print(f"Used RAM: {memory.used / (1024**3):.1f} GB ({memory.percent}%)")
    return memory.available / (1024**3)  # Return available GB

def load_model_with_optimizations():
    """Load the Qwen2.5-7b model from local weights with aggressive memory optimizations"""
    print("Checking system memory...")
    available_gb = check_memory()
    
    # Use local weights path
    model_path = "."  # Current directory contains the weights
    
    print(f"Attempting to load Qwen2.5-7B from local weights at: {os.path.abspath(model_path)}")
    
    try:
        print(f"Loading model from local weights...")
        
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Aggressive memory optimization for CPU
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Use half precision to save memory
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # Additional memory optimizations
            offload_folder="./offload",  # Offload to disk if needed
            offload_state_dict=True,
        )
        
        print(f"Successfully loaded Qwen2.5-7B from local weights")
        print("Model loaded with memory optimizations")
        
        # Check memory after loading
        print("\nMemory usage after loading:")
        check_memory()
        
        return model, tokenizer, "Qwen2.5-7B-Instruct (Local)"
        
    except Exception as e:
        print(f"Error loading local model: {e}")
        print("Trying fallback to smaller model...")
        
        # Fallback to smaller model from Hugging Face
        try:
            model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            print(f"Successfully loaded fallback model: {model_name}")
            return model, tokenizer, model_name
        except Exception as e2:
            print(f"Error loading fallback model: {e2}")
            return None, None, None

def generate_response(model, tokenizer, question, max_new_tokens=256):
    """Generate a response with memory monitoring"""
    
    print(f"\nMemory before generation:")
    check_memory()
    
    # Prepare the conversation
    messages = [
        {"role": "system", "content": "You are Qwen, a helpful AI assistant. Provide detailed and thoughtful answers."},
        {"role": "user", "content": question}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    model_inputs = tokenizer([text], return_tensors="pt")
    input_length = model_inputs.input_ids.shape[1]
    
    print(f"Input length: {input_length} tokens")
    print("Generating response...")
    
    start_time = time.time()
    
    # Set fixed RNG seed for reproducibility
    torch.manual_seed(42)
    
    # Generate with memory optimization
    with torch.no_grad():
        # Use smaller batch size and shorter sequences for memory efficiency
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,  # Enable KV cache for efficiency
        )
    
    generation_time = time.time() - start_time
    
    # Extract generated text
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    tokens_generated = len(generated_ids[0])
    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
    
    print(f"Generation completed in {generation_time:.2f} seconds")
    print(f"Generated {tokens_generated} tokens ({tokens_per_second:.1f} tokens/sec)")
    
    # Clean up
    del model_inputs, generated_ids
    gc.collect()
    
    return response

def main():
    """Main inference function"""
    
    print("="*80)
    print("QWEN2.5-7B MODEL INFERENCE - DOG FOOD TEST")
    print("Using Local Weights")
    print("="*80)
    
    # Load model
    model, tokenizer, model_name = load_model_with_optimizations()
    
    if model is None:
        print("Failed to load any model. Exiting.")
        return
    
    print(f"\nUsing model: {model_name}")
    
    # Dog food question only
    question = "Janet's dogs eat 2 pounds of dog food each day. If Janet buys a 50-pound bag of dog food, how many days will it last?"
    
    print(f"\n" + "="*60)
    print(f"DOG FOOD QUESTION:")
    print(question)
    print("\n" + "="*60)
    print("RESPONSE:")
    
    try:
        response = generate_response(model, tokenizer, question, max_new_tokens=256)
        print(response)
        
    except Exception as e:
        print(f"Error during generation: {e}")
        print("This might be due to insufficient memory for the 7B model.")
    
    print(f"\n" + "-"*60)
    
    # Clean up
    del model, tokenizer
    gc.collect()
    
    print(f"\nFinal memory state:")
    check_memory()
    
    print("\nDog food test completed!")

if __name__ == "__main__":
    main()