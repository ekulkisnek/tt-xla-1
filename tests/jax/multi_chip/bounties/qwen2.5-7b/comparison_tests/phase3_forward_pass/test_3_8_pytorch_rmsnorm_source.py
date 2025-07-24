#!/usr/bin/env python3
"""
Phase 3.8: PyTorch RMSNorm Source Investigation
==============================================
Investigate PyTorch's actual RMSNorm implementation to understand the differences.
"""

import sys
import os
import numpy as np
import time
import gc
import json
import math
import inspect
from pathlib import Path

def investigate_pytorch_rmsnorm_source():
    """Investigate PyTorch's RMSNorm source code"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print("Loading PyTorch model to inspect RMSNorm...")
        model_path = "../../weights"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model.eval()
        
        # Get the RMSNorm layer
        layer_0 = model.model.layers[0]
        input_layernorm = layer_0.input_layernorm
        
        print(f"RMSNorm class: {input_layernorm.__class__}")
        print(f"RMSNorm module: {input_layernorm.__class__.__module__}")
        
        # Get the source code
        try:
            source = inspect.getsource(input_layernorm.__class__)
            print("\n" + "=" * 70)
            print("PYTORCH RMSNORM SOURCE CODE")
            print("=" * 70)
            print(source)
        except:
            print("Could not get source code")
        
        # Get the forward method specifically
        try:
            forward_source = inspect.getsource(input_layernorm.forward)
            print("\n" + "=" * 70)
            print("PYTORCH RMSNORM FORWARD METHOD")
            print("=" * 70)
            print(forward_source)
        except:
            print("Could not get forward method source")
        
        # Test the actual computation step by step
        inputs = tokenizer("Hello", return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            embedding_output = model.model.embed_tokens(input_ids)
            
            print(f"\n" + "=" * 70)
            print("MANUAL RMSNORM REPLICATION")
            print("=" * 70)
            
            # Get parameters
            x = embedding_output
            weight = input_layernorm.weight
            eps = input_layernorm.variance_epsilon
            
            print(f"Input dtype: {x.dtype}")
            print(f"Weight dtype: {weight.dtype}")
            print(f"Eps: {eps}")
            
            # Try to replicate exactly what PyTorch does
            # Based on typical RMSNorm implementation:
            # 1. Convert to float for computation
            # 2. Compute variance
            # 3. Normalize 
            # 4. Scale
            # 5. Convert back to original dtype
            
            print(f"\nTesting different approaches:")
            
            # Approach 1: All in original dtype (bfloat16)
            x_bf16 = x
            variance_bf16 = x_bf16.pow(2).mean(-1, keepdim=True)
            rms_bf16 = torch.sqrt(variance_bf16 + eps)
            normalized_bf16 = x_bf16 / rms_bf16
            output_bf16 = normalized_bf16 * weight
            builtin_output = input_layernorm(x)
            diff_bf16 = torch.abs(output_bf16 - builtin_output).max()
            print(f"Approach 1 (all bf16): max diff = {diff_bf16.item():.6e}")
            
            # Approach 2: Computation in float32, then convert back
            x_f32 = x.float()
            weight_f32 = weight.float()
            variance_f32 = x_f32.pow(2).mean(-1, keepdim=True)
            rms_f32 = torch.sqrt(variance_f32 + eps)
            normalized_f32 = x_f32 / rms_f32
            output_f32 = (normalized_f32 * weight_f32).to(x.dtype)
            diff_f32 = torch.abs(output_f32 - builtin_output).max()
            print(f"Approach 2 (f32 then convert): max diff = {diff_f32.item():.6e}")
            
            # Approach 3: Mixed precision (what might PyTorch actually do)
            x_mixed = x
            variance_mixed = x_mixed.float().pow(2).mean(-1, keepdim=True)
            rms_mixed = torch.sqrt(variance_mixed + eps).to(x.dtype)
            normalized_mixed = x_mixed / rms_mixed
            output_mixed = normalized_mixed * weight
            diff_mixed = torch.abs(output_mixed - builtin_output).max()
            print(f"Approach 3 (mixed precision): max diff = {diff_mixed.item():.6e}")
            
            # Check if there's a specific implementation detail we're missing
            print(f"\nBuilt-in output sample: {builtin_output[0, 0, :5]}")
            print(f"Approach 1 output sample: {output_bf16[0, 0, :5]}")
            print(f"Approach 2 output sample: {output_f32[0, 0, :5]}")
            print(f"Approach 3 output sample: {output_mixed[0, 0, :5]}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main investigation function"""
    print("=" * 70)
    print("PHASE 3.8: PYTORCH RMSNORM SOURCE INVESTIGATION")
    print("=" * 70)
    
    success = investigate_pytorch_rmsnorm_source()
    
    print("\n" + "=" * 70)
    print("SOURCE INVESTIGATION COMPLETE")
    print("=" * 70)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 