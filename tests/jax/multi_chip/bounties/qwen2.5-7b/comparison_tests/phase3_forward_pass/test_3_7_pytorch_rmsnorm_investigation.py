#!/usr/bin/env python3
"""
Phase 3.7: PyTorch RMSNorm Investigation
========================================
Investigate how PyTorch's RMSNorm layer actually works vs manual implementation.
"""

import sys
import os
import numpy as np
import time
import gc
import json
import math
from pathlib import Path

def get_test_input():
    """Get test input for investigation"""
    return "Hello"  # Token 9707 - confirmed identical embedding

def investigate_pytorch_rmsnorm():
    """Deep investigation of PyTorch RMSNorm implementation"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print("Loading PyTorch model for RMSNorm investigation...")
        model_path = "../../weights"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model.eval()
        
        # Tokenize input
        inputs = tokenizer("Hello", return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        print(f"Input tokens: {input_ids.tolist()}")
        
        with torch.no_grad():
            # Get embedding
            embedding_output = model.model.embed_tokens(input_ids)
            
            # Get Layer 0 input layernorm
            layer_0 = model.model.layers[0]
            input_layernorm = layer_0.input_layernorm
            
            print(f"\nRMSNorm layer type: {type(input_layernorm)}")
            print(f"RMSNorm layer attributes: {dir(input_layernorm)}")
            
            # Check if it has variance_epsilon
            if hasattr(input_layernorm, 'variance_epsilon'):
                print(f"variance_epsilon: {input_layernorm.variance_epsilon}")
            if hasattr(input_layernorm, 'eps'):
                print(f"eps: {input_layernorm.eps}")
                
            # Check the weight
            weight = input_layernorm.weight
            print(f"Weight shape: {weight.shape}")
            print(f"Weight dtype: {weight.dtype}")
            
            # Test with different input dtypes
            print(f"\n--- TESTING DIFFERENT INPUT DTYPES ---")
            
            # Test 1: bfloat16 input (original)
            print(f"\nTest 1: bfloat16 input")
            x_bf16 = embedding_output  # This is bfloat16
            output_bf16 = input_layernorm(x_bf16)
            print(f"Input dtype: {x_bf16.dtype}")
            print(f"Output dtype: {output_bf16.dtype}")
            print(f"Output sample: {output_bf16[0, 0, :5].float()}")
            
            # Test 2: float32 input
            print(f"\nTest 2: float32 input")
            x_f32 = embedding_output.float()
            output_f32 = input_layernorm(x_f32)
            print(f"Input dtype: {x_f32.dtype}")
            print(f"Output dtype: {output_f32.dtype}")
            print(f"Output sample: {output_f32[0, 0, :5]}")
            
            # Compare outputs
            diff_dtype = torch.abs(output_bf16.float() - output_f32)
            print(f"bf16 vs f32 max diff: {diff_dtype.max().item():.6e}")
            
            # Manual implementation with different dtypes
            print(f"\n--- MANUAL IMPLEMENTATION TESTS ---")
            
            # Manual with bfloat16 (matching PyTorch input)
            print(f"\nManual bfloat16:")
            x = x_bf16
            variance = x.pow(2).mean(-1, keepdim=True)
            eps = 1e-6
            rms = torch.sqrt(variance + eps)
            normalized = x / rms
            manual_bf16 = normalized * weight
            print(f"Manual bf16 output dtype: {manual_bf16.dtype}")
            print(f"Manual bf16 output sample: {manual_bf16[0, 0, :5].float()}")
            
            # Manual with float32
            print(f"\nManual float32:")
            x = x_f32
            variance = x.pow(2).mean(-1, keepdim=True)
            eps = 1e-6
            rms = torch.sqrt(variance + eps)
            normalized = x / rms
            manual_f32 = normalized * weight.float()
            print(f"Manual f32 output dtype: {manual_f32.dtype}")
            print(f"Manual f32 output sample: {manual_f32[0, 0, :5]}")
            
            # Compare manual vs built-in
            print(f"\n--- COMPARISONS ---")
            diff_manual_bf16 = torch.abs(output_bf16.float() - manual_bf16.float())
            diff_manual_f32 = torch.abs(output_f32 - manual_f32)
            
            print(f"Built-in bf16 vs Manual bf16: {diff_manual_bf16.max().item():.6e}")
            print(f"Built-in f32 vs Manual f32: {diff_manual_f32.max().item():.6e}")
            
            # Test weight scaling specifically
            print(f"\n--- WEIGHT SCALING TEST ---")
            
            # Use the normalized result that we know is identical
            x = x_f32
            variance = x.pow(2).mean(-1, keepdim=True)
            eps = 1e-6
            rms = torch.sqrt(variance + eps)
            normalized = x / rms
            
            # Test different ways of scaling
            scale1 = normalized * weight.float()  # Regular multiplication
            scale2 = normalized * weight.float().unsqueeze(0).unsqueeze(0)  # Explicit broadcasting
            scale3 = torch.mul(normalized, weight.float())  # torch.mul
            
            print(f"Scale method 1 (regular *): {scale1[0, 0, :5]}")
            print(f"Scale method 2 (broadcast): {scale2[0, 0, :5]}")
            print(f"Scale method 3 (torch.mul): {scale3[0, 0, :5]}")
            
            print(f"Method 1 vs 2 diff: {torch.abs(scale1 - scale2).max().item():.6e}")
            print(f"Method 1 vs 3 diff: {torch.abs(scale1 - scale3).max().item():.6e}")
            
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Main investigation function"""
    print("=" * 70)
    print("PHASE 3.7: PYTORCH RMSNORM INVESTIGATION")
    print("=" * 70)
    
    success = investigate_pytorch_rmsnorm()
    
    print("\n" + "=" * 70)
    print("INVESTIGATION COMPLETE")
    print("=" * 70)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 