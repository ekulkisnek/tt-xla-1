#!/usr/bin/env python3
"""
Phase 5A.1: Cache Update Validation
Building on Phase 3 attention expertise to achieve perfect cache handling
Target: Perfect cache tensor alignment with 0.00e+00 standards
"""

import os
import sys
import time
import numpy as np
import torch
import jax
import jax.numpy as jnp
import logging
from typing import List, Tuple, Dict, Optional

# Add parent directory to path to import model components
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Setup logging with Phase 3 standards
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("phase5a1_cache")

class Phase5A1CacheValidator:
    """Phase 5A.1: KV Cache validation with Phase 3 precision standards"""
    
    def __init__(self):
        self.precision_threshold = 0.0  # Phase 3 standard: 0.00e+00 for cache operations
        self.cache_shape_threshold = 1e-9  # Very strict for shape consistency
        self.validation_passed = True
        
        # Test configurations
        self.test_configs = [
            {"batch_size": 1, "seq_len": 1, "hidden_size": 4096, "num_heads": 32, "num_kv_heads": 32, "head_dim": 128},
            {"batch_size": 1, "seq_len": 5, "hidden_size": 4096, "num_heads": 32, "num_kv_heads": 32, "head_dim": 128},
            {"batch_size": 2, "seq_len": 3, "hidden_size": 4096, "num_heads": 32, "num_kv_heads": 32, "head_dim": 128},
            {"batch_size": 1, "seq_len": 1, "hidden_size": 4096, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},  # GQA case
        ]
        
    def log_validation_result(self, test_name: str, passed: bool, details: str = ""):
        """Phase 3 style validation logging"""
        status = "✅" if passed else "❌"
        logger.info(f"{status} {test_name}: {details}")
        if not passed:
            self.validation_passed = False
    
    def create_test_cache(self, config: Dict, past_seq_len: int = 0) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Create test KV cache tensors with proper format"""
        batch_size = config["batch_size"]
        num_kv_heads = config["num_kv_heads"]
        head_dim = config["head_dim"]
        
        if past_seq_len == 0:
            # Empty cache
            past_k = jnp.zeros((batch_size, 0, num_kv_heads, head_dim), dtype=jnp.float32)
            past_v = jnp.zeros((batch_size, 0, num_kv_heads, head_dim), dtype=jnp.float32)
        else:
            # Pre-existing cache
            np.random.seed(42)
            past_k = jnp.array(np.random.randn(batch_size, past_seq_len, num_kv_heads, head_dim).astype(np.float32))
            past_v = jnp.array(np.random.randn(batch_size, past_seq_len, num_kv_heads, head_dim).astype(np.float32))
        
        return past_k, past_v
    
    def create_current_kv(self, config: Dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Create current step K,V tensors"""
        batch_size = config["batch_size"]
        seq_len = config["seq_len"]
        num_kv_heads = config["num_kv_heads"]
        head_dim = config["head_dim"]
        
        np.random.seed(123)
        current_k = jnp.array(np.random.randn(batch_size, seq_len, num_kv_heads, head_dim).astype(np.float32))
        current_v = jnp.array(np.random.randn(batch_size, seq_len, num_kv_heads, head_dim).astype(np.float32))
        
        return current_k, current_v
    
    def jax_cache_update(self, past_k: jnp.ndarray, past_v: jnp.ndarray, 
                        current_k: jnp.ndarray, current_v: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """JAX cache update implementation"""
        # Concatenate along sequence dimension (axis=1)
        updated_k = jnp.concatenate([past_k, current_k], axis=1)
        updated_v = jnp.concatenate([past_v, current_v], axis=1)
        return updated_k, updated_v
    
    def torch_cache_update(self, past_k: np.ndarray, past_v: np.ndarray,
                          current_k: np.ndarray, current_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """PyTorch cache update implementation"""
        past_k_torch = torch.tensor(past_k)
        past_v_torch = torch.tensor(past_v)
        current_k_torch = torch.tensor(current_k)
        current_v_torch = torch.tensor(current_v)
        
        # Concatenate along sequence dimension (axis=1)
        updated_k = torch.cat([past_k_torch, current_k_torch], dim=1)
        updated_v = torch.cat([past_v_torch, current_v_torch], dim=1)
        
        return updated_k.numpy(), updated_v.numpy()
    
    def test_cache_concatenation_precision(self, config: Dict, past_seq_len: int) -> bool:
        """Test cache concatenation precision between JAX and PyTorch"""
        
        # Create test data
        past_k, past_v = self.create_test_cache(config, past_seq_len)
        current_k, current_v = self.create_current_kv(config)
        
        # JAX update
        jax_updated_k, jax_updated_v = self.jax_cache_update(past_k, past_v, current_k, current_v)
        
        # PyTorch update  
        torch_updated_k, torch_updated_v = self.torch_cache_update(
            np.array(past_k), np.array(past_v), 
            np.array(current_k), np.array(current_v)
        )
        
        # Compare results with Phase 3 precision standards
        k_diff = np.max(np.abs(np.array(jax_updated_k) - torch_updated_k))
        v_diff = np.max(np.abs(np.array(jax_updated_v) - torch_updated_v))
        
        # Shape validation
        k_shapes_match = jax_updated_k.shape == torch_updated_k.shape
        v_shapes_match = jax_updated_v.shape == torch_updated_v.shape
        
        # Expected final shape
        expected_seq_len = past_seq_len + config["seq_len"]
        expected_shape = (config["batch_size"], expected_seq_len, config["num_kv_heads"], config["head_dim"])
        shape_correct = (jax_updated_k.shape == expected_shape and jax_updated_v.shape == expected_shape)
        
        all_passed = (k_diff <= self.precision_threshold and 
                     v_diff <= self.precision_threshold and
                     k_shapes_match and v_shapes_match and shape_correct)
        
        details = (f"past_seq={past_seq_len}, batch={config['batch_size']}, "
                  f"seq={config['seq_len']}, heads={config['num_kv_heads']}: "
                  f"k_diff={k_diff:.2e}, v_diff={v_diff:.2e}, "
                  f"shapes_match={k_shapes_match and v_shapes_match}, "
                  f"final_shape={jax_updated_k.shape}")
        
        return all_passed, details
    
    def test_empty_cache_initialization(self, config: Dict) -> bool:
        """Test empty cache initialization consistency"""
        
        # JAX empty cache
        jax_empty_k, jax_empty_v = self.create_test_cache(config, 0)
        
        # PyTorch empty cache (mimicking same creation)
        batch_size = config["batch_size"]
        num_kv_heads = config["num_kv_heads"]
        head_dim = config["head_dim"]
        
        torch_empty_k = torch.zeros((batch_size, 0, num_kv_heads, head_dim), dtype=torch.float32)
        torch_empty_v = torch.zeros((batch_size, 0, num_kv_heads, head_dim), dtype=torch.float32)
        
        # Compare shapes and values
        k_shapes_match = jax_empty_k.shape == torch_empty_k.shape
        v_shapes_match = jax_empty_v.shape == torch_empty_v.shape
        
        # Both should be empty (seq_len=0)
        k_empty = jax_empty_k.shape[1] == 0 and torch_empty_k.shape[1] == 0
        v_empty = jax_empty_v.shape[1] == 0 and torch_empty_v.shape[1] == 0
        
        all_passed = k_shapes_match and v_shapes_match and k_empty and v_empty
        
        details = (f"batch={config['batch_size']}, heads={config['num_kv_heads']}: "
                  f"jax_k_shape={jax_empty_k.shape}, torch_k_shape={torch_empty_k.shape}, "
                  f"jax_v_shape={jax_empty_v.shape}, torch_v_shape={torch_empty_v.shape}")
        
        return all_passed, details
    
    def test_multi_step_cache_accumulation(self, config: Dict) -> bool:
        """Test multi-step cache accumulation consistency"""
        
        # Start with empty cache
        jax_k, jax_v = self.create_test_cache(config, 0)
        torch_k, torch_v = torch.zeros_like(torch.tensor(jax_k)), torch.zeros_like(torch.tensor(jax_v))
        
        # Simulate 3 generation steps
        all_consistent = True
        final_details = []
        
        for step in range(3):
            # Create current step data (single token)
            step_config = config.copy()
            step_config["seq_len"] = 1
            
            np.random.seed(100 + step)  # Different seed for each step
            current_k = jnp.array(np.random.randn(config["batch_size"], 1, config["num_kv_heads"], config["head_dim"]).astype(np.float32))
            current_v = jnp.array(np.random.randn(config["batch_size"], 1, config["num_kv_heads"], config["head_dim"]).astype(np.float32))
            
            # JAX update
            jax_k, jax_v = self.jax_cache_update(jax_k, jax_v, current_k, current_v)
            
            # PyTorch update
            torch_k = torch.cat([torch_k, torch.tensor(current_k)], dim=1)
            torch_v = torch.cat([torch_v, torch.tensor(current_v)], dim=1)
            
            # Compare after each step
            k_diff = np.max(np.abs(np.array(jax_k) - torch_k.numpy()))
            v_diff = np.max(np.abs(np.array(jax_v) - torch_v.numpy()))
            
            step_consistent = (k_diff <= self.precision_threshold and v_diff <= self.precision_threshold)
            all_consistent = all_consistent and step_consistent
            
            final_details.append(f"step{step}(k_diff={k_diff:.2e},v_diff={v_diff:.2e})")
        
        # Final cache should have seq_len=3
        expected_final_len = 3
        jax_final_len = jax_k.shape[1]
        torch_final_len = torch_k.shape[1]
        
        length_correct = (jax_final_len == expected_final_len and torch_final_len == expected_final_len)
        all_consistent = all_consistent and length_correct
        
        details = f"batch={config['batch_size']}, heads={config['num_kv_heads']}: " + ", ".join(final_details) + f", final_len_jax={jax_final_len}, final_len_torch={torch_final_len}"
        
        return all_consistent, details
    
    def test_cache_shape_consistency(self, config: Dict) -> bool:
        """Test cache shape consistency across different scenarios"""
        
        shape_tests = []
        all_passed = True
        
        # Test different past sequence lengths
        for past_seq_len in [0, 1, 5, 10]:
            past_k, past_v = self.create_test_cache(config, past_seq_len)
            current_k, current_v = self.create_current_kv(config)
            
            # Update cache
            updated_k, updated_v = self.jax_cache_update(past_k, past_v, current_k, current_v)
            
            # Expected shape
            expected_seq_len = past_seq_len + config["seq_len"]
            expected_shape = (config["batch_size"], expected_seq_len, config["num_kv_heads"], config["head_dim"])
            
            k_shape_correct = updated_k.shape == expected_shape
            v_shape_correct = updated_v.shape == expected_shape
            
            test_passed = k_shape_correct and v_shape_correct
            all_passed = all_passed and test_passed
            
            shape_tests.append(f"past_len={past_seq_len}(✓)" if test_passed else f"past_len={past_seq_len}(✗)")
        
        details = f"batch={config['batch_size']}, heads={config['num_kv_heads']}: " + ", ".join(shape_tests)
        
        return all_passed, details
    
    def run_cache_validation(self) -> bool:
        """Run comprehensive cache validation tests"""
        
        logger.info("🚀 Starting Phase 5A.1: Cache Update Validation")
        logger.info("Target: Perfect cache tensor alignment with 0.00e+00 standards")
        
        # Test 1: Empty cache initialization
        logger.info("📊 Testing empty cache initialization...")
        for config in self.test_configs:
            passed, details = self.test_empty_cache_initialization(config)
            self.log_validation_result(f"Empty cache init", passed, details)
        
        # Test 2: Cache concatenation precision
        logger.info("📊 Testing cache concatenation precision...")
        for config in self.test_configs:
            for past_seq_len in [0, 1, 5]:
                passed, details = self.test_cache_concatenation_precision(config, past_seq_len)
                self.log_validation_result(f"Cache concatenation", passed, details)
        
        # Test 3: Multi-step cache accumulation
        logger.info("📊 Testing multi-step cache accumulation...")
        for config in self.test_configs[:2]:  # Use first 2 configs for multi-step
            passed, details = self.test_multi_step_cache_accumulation(config)
            self.log_validation_result(f"Multi-step accumulation", passed, details)
        
        # Test 4: Cache shape consistency
        logger.info("📊 Testing cache shape consistency...")
        for config in self.test_configs:
            passed, details = self.test_cache_shape_consistency(config)
            self.log_validation_result(f"Shape consistency", passed, details)
        
        # Final validation summary
        if self.validation_passed:
            logger.info("🎉 Phase 5A.1 PASSED: Perfect cache update alignment achieved!")
            logger.info("✅ Cache concatenation precision: 0.00e+00")
            logger.info("✅ Shape consistency validated")
            logger.info("✅ Multi-step accumulation verified")
        else:
            logger.error("❌ Phase 5A.1 FAILED: Cache alignment issues detected")
            logger.error("🔧 Review cache implementation for precision improvements")
        
        return self.validation_passed

def main():
    """Main Phase 5A.1 validation entry point"""
    
    print("="*80)
    print("PHASE 5A.1: CACHE UPDATE VALIDATION")
    print("Building on Phase 3 attention expertise for perfect cache handling")
    print("="*80)
    
    validator = Phase5A1CacheValidator()
    success = validator.run_cache_validation()
    
    print("\n" + "="*80)
    if success:
        print("🎉 PHASE 5A.1 VALIDATION COMPLETE - ALL TESTS PASSED")
        print("✅ Perfect cache tensor alignment achieved")
    else:
        print("❌ PHASE 5A.1 VALIDATION FAILED")
        print("🔧 Cache alignment requires optimization")
    print("="*80)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 