#!/usr/bin/env python3
"""
Phase 1.1: Config File Loading Comparison
=========================================
Tests that both PyTorch and JAX load the same configuration parameters from config.json.
This is the foundation for ensuring model architecture matches.
"""

import json
import sys
import os
from pathlib import Path

# Add the current directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def load_pytorch_config():
    """Load config using PyTorch/Transformers approach"""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("../../weights")
        
        # Extract key parameters
        pytorch_config = {
            'hidden_size': config.hidden_size,
            'num_attention_heads': config.num_attention_heads,
            'num_key_value_heads': getattr(config, 'num_key_value_heads', config.num_attention_heads),
            'num_hidden_layers': config.num_hidden_layers,
            'intermediate_size': config.intermediate_size,
            'vocab_size': config.vocab_size,
            'max_position_embeddings': config.max_position_embeddings,
            'rope_theta': getattr(config, 'rope_theta', 10000.0),
            'rms_norm_eps': getattr(config, 'rms_norm_eps', 1e-6),
            'tie_word_embeddings': bool(getattr(config, 'tie_word_embeddings', False)),
            'model_type': config.model_type,
            'torch_dtype': str(config.torch_dtype).replace('torch.', '') if hasattr(config, 'torch_dtype') else None,
        }
        
        return pytorch_config, None
        
    except Exception as e:
        return None, str(e)

def load_jax_config():
    """Load config using direct JSON loading (JAX approach)"""
    try:
        config_path = "../../weights/config.json"
        with open(config_path, 'r') as f:
            raw_config = json.load(f)
        
        # Extract same key parameters
        jax_config = {
            'hidden_size': raw_config.get('hidden_size'),
            'num_attention_heads': raw_config.get('num_attention_heads'),
            'num_key_value_heads': raw_config.get('num_key_value_heads', raw_config.get('num_attention_heads')),
            'num_hidden_layers': raw_config.get('num_hidden_layers'),
            'intermediate_size': raw_config.get('intermediate_size'),
            'vocab_size': raw_config.get('vocab_size'),
            'max_position_embeddings': raw_config.get('max_position_embeddings'),
            'rope_theta': raw_config.get('rope_theta', 10000.0),
            'rms_norm_eps': raw_config.get('rms_norm_eps', 1e-6),
            'tie_word_embeddings': bool(raw_config.get('tie_word_embeddings', False)),
            'model_type': raw_config.get('model_type'),
            'torch_dtype': raw_config.get('torch_dtype'),
        }
        
        return jax_config, None
        
    except Exception as e:
        return None, str(e)

def compare_configs():
    """Compare configurations and report differences"""
    print("=" * 60)
    print("PHASE 1.1: CONFIG FILE LOADING COMPARISON")
    print("=" * 60)
    
    # Load both configs
    pytorch_config, pytorch_error = load_pytorch_config()
    jax_config, jax_error = load_jax_config()
    
    # Check for loading errors
    if pytorch_error:
        print(f"❌ PyTorch config loading FAILED: {pytorch_error}")
        return False
    
    if jax_error:
        print(f"❌ JAX config loading FAILED: {jax_error}")
        return False
    
    print("✅ Both configs loaded successfully")
    print()
    
    # Compare each parameter
    all_match = True
    max_key_len = max(len(k) for k in pytorch_config.keys())
    
    print("Parameter Comparison:")
    print("-" * 60)
    
    for key in sorted(pytorch_config.keys()):
        pytorch_val = pytorch_config[key]
        jax_val = jax_config[key]
        
        match_status = "✅" if pytorch_val == jax_val else "❌"
        if pytorch_val != jax_val:
            all_match = False
        
        print(f"{match_status} {key:<{max_key_len}} | PyTorch: {pytorch_val:<15} | JAX: {jax_val}")
    
    print("-" * 60)
    
    if all_match:
        print("✅ ALL CONFIG PARAMETERS MATCH!")
        print("Phase 1.1 PASSED - Ready for Phase 1.2")
    else:
        print("❌ CONFIG MISMATCHES DETECTED")
        print("Must fix config loading before proceeding")
    
    return all_match

if __name__ == "__main__":
    success = compare_configs()
    sys.exit(0 if success else 1) 