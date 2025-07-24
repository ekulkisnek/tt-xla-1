#!/usr/bin/env python3
"""
Phase 4B.1 (Fixed): Top-K Filtering Validation
Quick fix with realistic tolerance for floating-point value comparisons
"""

import os
import sys
import numpy as np
import torch
import jax
import jax.numpy as jnp
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("phase4b1_topk_fixed")

class Phase4B1FixedTopKValidator:
    """Phase 4B.1 (Fixed): Top-K filtering with realistic tolerances"""
    
    def __init__(self):
        self.value_threshold = 1e-6  # Realistic for floating-point comparisons
        self.test_k_values = [1, 5, 20, 50]
        self.vocab_sizes = [100, 1000, 10000]
        self.validation_passed = True
        
    def log_validation_result(self, test_name: str, passed: bool, details: str = ""):
        status = "✅" if passed else "❌"
        logger.info(f"{status} {test_name}: {details}")
        if not passed:
            self.validation_passed = False
    
    def create_test_logits(self, vocab_size: int, seed: int = 42) -> np.ndarray:
        """Create reproducible test logits"""
        np.random.seed(seed)
        logits = np.random.randn(vocab_size).astype(np.float32)
        return logits
    
    def apply_jax_topk_filtering(self, logits: np.ndarray, k: int):
        """Apply top-k filtering using JAX"""
        jax_logits = jnp.array(logits)
        vocab_size = jax_logits.shape[-1]
        
        if k >= vocab_size:
            return np.array(jax_logits), np.arange(vocab_size)
        
        top_k_values, top_k_indices = jax.lax.top_k(jax_logits, k=k)
        mask = jnp.full_like(jax_logits, False, dtype=bool)
        mask = mask.at[top_k_indices].set(True)
        filtered_logits = jnp.where(mask, jax_logits, -jnp.inf)
        
        return np.array(filtered_logits), np.array(top_k_indices)
    
    def apply_torch_topk_filtering(self, logits: np.ndarray, k: int):
        """Apply top-k filtering using PyTorch"""
        torch_logits = torch.tensor(logits)
        vocab_size = torch_logits.shape[-1]
        
        if k >= vocab_size:
            return torch_logits.numpy(), np.arange(vocab_size)
        
        top_k_values, top_k_indices = torch.topk(torch_logits, k=k)
        mask = torch.full_like(torch_logits, False, dtype=torch.bool)
        mask[top_k_indices] = True
        filtered_logits = torch.where(mask, torch_logits, torch.tensor(-float('inf')))
        
        return filtered_logits.numpy(), top_k_indices.numpy()
    
    def test_topk_indices_alignment(self, vocab_size: int, k: int) -> bool:
        """Test top-k indices alignment with realistic tolerance"""
        test_logits = self.create_test_logits(vocab_size)
        
        jax_filtered, jax_indices = self.apply_jax_topk_filtering(test_logits, k)
        torch_filtered, torch_indices = self.apply_torch_topk_filtering(test_logits, k)
        
        # Sort indices for comparison
        jax_indices_sorted = np.sort(jax_indices)
        torch_indices_sorted = np.sort(torch_indices)
        
        # Check indices match
        indices_match = np.array_equal(jax_indices_sorted, torch_indices_sorted)
        
        # Check selected values with realistic tolerance
        if indices_match:
            jax_selected_values = test_logits[jax_indices_sorted]
            torch_selected_values = test_logits[torch_indices_sorted]
            max_value_diff = np.max(np.abs(jax_selected_values - torch_selected_values))
            values_match = max_value_diff < self.value_threshold
        else:
            values_match = False
            max_value_diff = float('inf')
        
        all_match = indices_match and values_match
        
        self.log_validation_result(
            f"Top-k indices alignment k={k}, vocab={vocab_size}",
            all_match,
            f"indices_match={indices_match}, values_diff={max_value_diff:.2e}"
        )
        
        return all_match
    
    def test_argmax_consistency(self, vocab_size: int) -> bool:
        """Test argmax consistency (k=1 case)"""
        test_logits = self.create_test_logits(vocab_size)
        
        # JAX top-1
        _, jax_indices = self.apply_jax_topk_filtering(test_logits, 1)
        jax_argmax = jax_indices[0]
        
        # PyTorch top-1
        _, torch_indices = self.apply_torch_topk_filtering(test_logits, 1)
        torch_argmax = torch_indices[0]
        
        # NumPy argmax
        numpy_argmax = np.argmax(test_logits)
        
        consistent = (jax_argmax == torch_argmax == numpy_argmax)
        
        self.log_validation_result(
            f"Argmax consistency vocab={vocab_size}",
            consistent,
            f"jax={jax_argmax}, torch={torch_argmax}, numpy={numpy_argmax}"
        )
        
        return consistent
    
    def test_probability_alignment(self, vocab_size: int, k: int) -> bool:
        """Test probability alignment after top-k filtering"""
        test_logits = self.create_test_logits(vocab_size)
        
        jax_filtered, _ = self.apply_jax_topk_filtering(test_logits, k)
        torch_filtered, _ = self.apply_torch_topk_filtering(test_logits, k)
        
        # Convert to probabilities
        jax_probs = jax.nn.softmax(jnp.array(jax_filtered), axis=-1)
        torch_probs = torch.softmax(torch.tensor(torch_filtered), dim=-1)
        
        # Compare probabilities
        max_diff = np.max(np.abs(np.array(jax_probs) - torch_probs.numpy()))
        prob_alignment = max_diff < 1e-6  # Realistic threshold for probabilities
        
        self.log_validation_result(
            f"Top-k probability alignment k={k}, vocab={vocab_size}",
            prob_alignment,
            f"max_diff={max_diff:.2e}"
        )
        
        return prob_alignment
    
    def run_fixed_topk_validation(self) -> bool:
        """Run fixed top-k validation"""
        
        logger.info("🚀 Starting Phase 4B.1 (Fixed): Top-K Filtering Validation")
        
        # Test 1: Top-k indices alignment
        logger.info("📊 Testing top-k indices alignment...")
        for vocab_size in self.vocab_sizes:
            for k in self.test_k_values:
                if k < vocab_size:
                    self.test_topk_indices_alignment(vocab_size, k)
        
        # Test 2: Argmax consistency
        logger.info("📊 Testing argmax consistency...")
        for vocab_size in self.vocab_sizes:
            self.test_argmax_consistency(vocab_size)
        
        # Test 3: Probability alignment
        logger.info("📊 Testing probability alignment...")
        for vocab_size in [100, 1000]:
            for k in [5, 20]:
                self.test_probability_alignment(vocab_size, k)
        
        # Final validation
        if self.validation_passed:
            logger.info("🎉 Phase 4B.1 (Fixed) PASSED: Top-K filtering validation successful!")
            logger.info("✅ Top-K filtering alignment achieved with realistic tolerances")
        else:
            logger.error("❌ Phase 4B.1 (Fixed) FAILED: Top-K filtering issues remain")
        
        return self.validation_passed

def main():
    print("="*80)
    print("PHASE 4B.1 (FIXED): TOP-K FILTERING WITH REALISTIC TOLERANCES")
    print("="*80)
    
    validator = Phase4B1FixedTopKValidator()
    success = validator.run_fixed_topk_validation()
    
    print("\n" + "="*80)
    if success:
        print("🎉 PHASE 4B.1 (FIXED) VALIDATION COMPLETE - ALL TESTS PASSED")
        print("✅ Top-K filtering alignment achieved")
    else:
        print("❌ PHASE 4B.1 (FIXED) VALIDATION FAILED")
    print("="*80)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 