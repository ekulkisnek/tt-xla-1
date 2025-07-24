#!/usr/bin/env python3
"""
Phase 4B.1: Top-K Filtering Validation
Building on Phase 3 methodology to achieve perfect top_k implementation alignment
Target: Identical token selection for all k values with 0.00e+00 standards
"""

import os
import sys
import time
import numpy as np
import torch
import jax
import jax.numpy as jnp
import logging
from typing import List, Tuple, Dict

# Setup logging with Phase 3 standards
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("phase4b1_topk")

class Phase4B1TopKValidator:
    """Phase 4B.1: Top-K filtering validation using Phase 3 methodology"""
    
    def __init__(self, precision_threshold: float = 0.0):
        """Initialize with Phase 3 precision standards"""
        self.precision_threshold = precision_threshold  # 0.00e+00 target
        self.test_k_values = [1, 5, 10, 20, 50, 100, 200]
        self.vocab_sizes = [100, 1000, 10000, 32000, 152064]  # Including Qwen vocab size
        self.validation_passed = True
        
    def log_validation_result(self, test_name: str, passed: bool, details: str = ""):
        """Phase 3 style validation logging"""
        status = "✅" if passed else "❌"
        logger.info(f"{status} {test_name}: {details}")
        if not passed:
            self.validation_passed = False
    
    def create_test_logits(self, vocab_size: int, seed: int = 42, add_ties: bool = False) -> np.ndarray:
        """Create reproducible test logits with optional ties for validation"""
        np.random.seed(seed)
        logits = np.random.randn(vocab_size).astype(np.float32)
        
        if add_ties and vocab_size > 10:
            # Add intentional ties to test tie-breaking behavior
            logits[5:8] = 1.5  # Three identical values
            logits[15:17] = -0.5  # Two identical values
        
        return logits
    
    def apply_jax_topk_filtering(self, logits: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply top-k filtering using JAX implementation"""
        jax_logits = jnp.array(logits)
        vocab_size = jax_logits.shape[-1]
        
        if k <= 0:
            # No filtering
            return np.array(jax_logits), np.arange(vocab_size), np.ones_like(logits, dtype=bool)
        
        if k >= vocab_size:
            # Keep all tokens
            return np.array(jax_logits), np.arange(vocab_size), np.ones_like(logits, dtype=bool)
        
        # Get top-k values and indices
        top_k_values, top_k_indices = jax.lax.top_k(jax_logits, k=k)
        
        # Create mask for top-k tokens
        mask = jnp.full_like(jax_logits, False, dtype=bool)
        mask = mask.at[top_k_indices].set(True)
        
        # Set non-top-k logits to -inf
        filtered_logits = jnp.where(mask, jax_logits, -jnp.inf)
        
        return np.array(filtered_logits), np.array(top_k_indices), np.array(mask)
    
    def apply_torch_topk_filtering(self, logits: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply top-k filtering using PyTorch implementation"""
        torch_logits = torch.tensor(logits)
        vocab_size = torch_logits.shape[-1]
        
        if k <= 0:
            # No filtering
            return torch_logits.numpy(), np.arange(vocab_size), np.ones_like(logits, dtype=bool)
        
        if k >= vocab_size:
            # Keep all tokens
            return torch_logits.numpy(), np.arange(vocab_size), np.ones_like(logits, dtype=bool)
        
        # Get top-k values and indices
        top_k_values, top_k_indices = torch.topk(torch_logits, k=k)
        
        # Create mask for top-k tokens
        mask = torch.full_like(torch_logits, False, dtype=torch.bool)
        mask[top_k_indices] = True
        
        # Set non-top-k logits to -inf
        filtered_logits = torch.where(mask, torch_logits, torch.tensor(-float('inf')))
        
        return filtered_logits.numpy(), top_k_indices.numpy(), mask.numpy()
    
    def validate_topk_indices_alignment(self, vocab_size: int, k: int, add_ties: bool = False) -> bool:
        """Validate that top-k indices are identical between frameworks"""
        
        test_logits = self.create_test_logits(vocab_size, add_ties=add_ties)
        
        # Apply top-k filtering
        jax_filtered, jax_indices, jax_mask = self.apply_jax_topk_filtering(test_logits, k)
        torch_filtered, torch_indices, torch_mask = self.apply_torch_topk_filtering(test_logits, k)
        
        # Sort indices for comparison (order might differ but should be same set)
        jax_indices_sorted = np.sort(jax_indices)
        torch_indices_sorted = np.sort(torch_indices)
        
        # Check if indices are identical
        indices_match = np.array_equal(jax_indices_sorted, torch_indices_sorted)
        
        # Check if masks are identical
        masks_match = np.array_equal(jax_mask, torch_mask)
        
        # Check values at selected indices
        values_match = True
        if indices_match:
            jax_selected_values = test_logits[jax_indices_sorted]
            torch_selected_values = test_logits[torch_indices_sorted]
            max_value_diff = np.max(np.abs(jax_selected_values - torch_selected_values))
            values_match = max_value_diff < self.precision_threshold
        
        all_match = indices_match and masks_match and values_match
        
        self.log_validation_result(
            f"Top-k indices k={k}, vocab={vocab_size}, ties={add_ties}",
            all_match,
            f"indices_match={indices_match}, masks_match={masks_match}, values_match={values_match}"
        )
        
        return all_match
    
    def validate_topk_filtering_precision(self, vocab_size: int, k: int) -> bool:
        """Validate top-k filtering precision with Phase 3 standards"""
        
        test_logits = self.create_test_logits(vocab_size)
        
        # Apply top-k filtering
        jax_filtered, jax_indices, jax_mask = self.apply_jax_topk_filtering(test_logits, k)
        torch_filtered, torch_indices, torch_mask = self.apply_torch_topk_filtering(test_logits, k)
        
        # Compare filtered logits
        # Note: -inf values might differ slightly between frameworks
        # Focus on non-infinite values
        jax_finite_mask = np.isfinite(jax_filtered)
        torch_finite_mask = np.isfinite(torch_filtered)
        
        finite_masks_match = np.array_equal(jax_finite_mask, torch_finite_mask)
        
        if finite_masks_match:
            jax_finite_values = jax_filtered[jax_finite_mask]
            torch_finite_values = torch_filtered[torch_finite_mask]
            
            if len(jax_finite_values) > 0 and len(torch_finite_values) > 0:
                max_diff = np.max(np.abs(jax_finite_values - torch_finite_values))
                mean_diff = np.mean(np.abs(jax_finite_values - torch_finite_values))
                precision_ok = max_diff < self.precision_threshold
            else:
                max_diff = 0.0
                mean_diff = 0.0
                precision_ok = True
        else:
            precision_ok = False
            max_diff = float('inf')
            mean_diff = float('inf')
        
        self.log_validation_result(
            f"Top-k filtering precision k={k}, vocab={vocab_size}",
            precision_ok and finite_masks_match,
            f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, finite_masks_match={finite_masks_match}"
        )
        
        return precision_ok and finite_masks_match
    
    def test_topk_edge_cases(self, vocab_size: int) -> bool:
        """Test top-k edge cases using Phase 3 methodology"""
        
        edge_cases_passed = True
        test_logits = self.create_test_logits(vocab_size)
        
        # Edge case 1: k=0 (should keep all tokens)
        jax_filtered_0, _, jax_mask_0 = self.apply_jax_topk_filtering(test_logits, 0)
        torch_filtered_0, _, torch_mask_0 = self.apply_torch_topk_filtering(test_logits, 0)
        
        case1_ok = (np.allclose(jax_filtered_0, test_logits, atol=self.precision_threshold) and
                   np.allclose(torch_filtered_0, test_logits, atol=self.precision_threshold))
        
        if not case1_ok:
            edge_cases_passed = False
            self.log_validation_result(f"Top-k edge case k=0, vocab={vocab_size}", False, "k=0 should preserve all logits")
        
        # Edge case 2: k=1 (should keep only maximum)
        jax_filtered_1, jax_indices_1, _ = self.apply_jax_topk_filtering(test_logits, 1)
        torch_filtered_1, torch_indices_1, _ = self.apply_torch_topk_filtering(test_logits, 1)
        
        true_argmax = np.argmax(test_logits)
        case2_ok = (jax_indices_1[0] == true_argmax and torch_indices_1[0] == true_argmax)
        
        if not case2_ok:
            edge_cases_passed = False
            self.log_validation_result(
                f"Top-k edge case k=1, vocab={vocab_size}", 
                False, 
                f"jax_idx={jax_indices_1[0]}, torch_idx={torch_indices_1[0]}, true_argmax={true_argmax}"
            )
        
        # Edge case 3: k >= vocab_size (should keep all tokens)
        k_large = vocab_size + 10
        jax_filtered_large, _, jax_mask_large = self.apply_jax_topk_filtering(test_logits, k_large)
        torch_filtered_large, _, torch_mask_large = self.apply_torch_topk_filtering(test_logits, k_large)
        
        case3_ok = (np.allclose(jax_filtered_large, test_logits, atol=self.precision_threshold) and
                   np.allclose(torch_filtered_large, test_logits, atol=self.precision_threshold))
        
        if not case3_ok:
            edge_cases_passed = False
            self.log_validation_result(f"Top-k edge case k>vocab_size, vocab={vocab_size}", False, "Large k should preserve all logits")
        
        if edge_cases_passed:
            self.log_validation_result(f"Top-k edge cases vocab={vocab_size}", True, "All edge cases passed")
        
        return edge_cases_passed
    
    def validate_topk_tie_breaking(self, vocab_size: int, k: int) -> bool:
        """Validate consistent tie-breaking behavior between frameworks"""
        
        # Create logits with intentional ties
        test_logits = self.create_test_logits(vocab_size, add_ties=True)
        
        # Apply top-k filtering multiple times to check consistency
        results_jax = []
        results_torch = []
        
        for run in range(3):  # Multiple runs to check determinism
            jax_filtered, jax_indices, _ = self.apply_jax_topk_filtering(test_logits, k)
            torch_filtered, torch_indices, _ = self.apply_torch_topk_filtering(test_logits, k)
            
            results_jax.append(np.sort(jax_indices))
            results_torch.append(np.sort(torch_indices))
        
        # Check JAX consistency
        jax_consistent = all(np.array_equal(results_jax[0], result) for result in results_jax[1:])
        
        # Check PyTorch consistency
        torch_consistent = all(np.array_equal(results_torch[0], result) for result in results_torch[1:])
        
        # Check cross-framework consistency
        cross_consistent = np.array_equal(results_jax[0], results_torch[0])
        
        tie_breaking_ok = jax_consistent and torch_consistent and cross_consistent
        
        self.log_validation_result(
            f"Top-k tie breaking k={k}, vocab={vocab_size}",
            tie_breaking_ok,
            f"jax_consistent={jax_consistent}, torch_consistent={torch_consistent}, cross_consistent={cross_consistent}"
        )
        
        return tie_breaking_ok
    
    def validate_topk_probability_distribution(self, vocab_size: int, k: int) -> bool:
        """Validate that probability distributions after top-k are identical"""
        
        test_logits = self.create_test_logits(vocab_size)
        
        # Apply top-k filtering
        jax_filtered, _, _ = self.apply_jax_topk_filtering(test_logits, k)
        torch_filtered, _, _ = self.apply_torch_topk_filtering(test_logits, k)
        
        # Convert to probabilities using softmax
        jax_probs = jax.nn.softmax(jnp.array(jax_filtered), axis=-1)
        torch_probs = torch.softmax(torch.tensor(torch_filtered), dim=-1)
        
        # Compare probabilities
        jax_probs_np = np.array(jax_probs)
        torch_probs_np = torch_probs.numpy()
        
        max_diff = np.max(np.abs(jax_probs_np - torch_probs_np))
        
        # Check probability sum (should be 1.0)
        jax_sum = np.sum(jax_probs_np)
        torch_sum = np.sum(torch_probs_np)
        
        prob_precision_ok = max_diff < 1e-6  # Slightly relaxed for probability comparison
        sum_ok = abs(jax_sum - 1.0) < 1e-6 and abs(torch_sum - 1.0) < 1e-6
        
        self.log_validation_result(
            f"Top-k probability distribution k={k}, vocab={vocab_size}",
            prob_precision_ok and sum_ok,
            f"max_diff={max_diff:.2e}, jax_sum={jax_sum:.6f}, torch_sum={torch_sum:.6f}"
        )
        
        return prob_precision_ok and sum_ok
    
    def run_comprehensive_topk_validation(self) -> bool:
        """Run comprehensive top-k validation using Phase 3 methodology"""
        
        logger.info("🚀 Starting Phase 4B.1: Top-K Filtering Validation")
        logger.info(f"Target precision: {self.precision_threshold:.2e}")
        
        # Test 1: Top-k indices alignment
        logger.info("📊 Testing top-k indices alignment...")
        for vocab_size in self.vocab_sizes:
            for k in self.test_k_values:
                if k < vocab_size:  # Only test meaningful k values
                    self.validate_topk_indices_alignment(vocab_size, k)
                    if vocab_size <= 1000:  # Test ties for smaller vocabularies
                        self.validate_topk_indices_alignment(vocab_size, k, add_ties=True)
        
        # Test 2: Top-k filtering precision
        logger.info("📊 Testing top-k filtering precision...")
        for vocab_size in [1000, 10000, 32000]:  # Representative sizes
            for k in [1, 5, 20, 50]:
                self.validate_topk_filtering_precision(vocab_size, k)
        
        # Test 3: Edge cases
        logger.info("📊 Testing top-k edge cases...")
        for vocab_size in [100, 1000, 10000]:
            self.test_topk_edge_cases(vocab_size)
        
        # Test 4: Tie-breaking behavior
        logger.info("📊 Testing top-k tie-breaking behavior...")
        for vocab_size in [100, 1000]:
            for k in [5, 10, 20]:
                if k < vocab_size:
                    self.validate_topk_tie_breaking(vocab_size, k)
        
        # Test 5: Probability distribution alignment
        logger.info("📊 Testing top-k probability distribution alignment...")
        for vocab_size in [100, 1000]:
            for k in [5, 20]:
                self.validate_topk_probability_distribution(vocab_size, k)
        
        # Final validation summary
        if self.validation_passed:
            logger.info("🎉 Phase 4B.1 PASSED: Top-K filtering validation successful!")
            logger.info("✅ Achieved identical token selection for all k values")
        else:
            logger.error("❌ Phase 4B.1 FAILED: Top-K filtering alignment issues detected")
            logger.error("🔧 Review top-k implementation for precision improvements")
        
        return self.validation_passed

def main():
    """Main Phase 4B.1 validation entry point"""
    
    print("="*80)
    print("PHASE 4B.1: TOP-K FILTERING VALIDATION")
    print("Building on Phase 3 methodology for perfect top-k implementation alignment")
    print("="*80)
    
    # Initialize validator with Phase 3 precision standards
    validator = Phase4B1TopKValidator(precision_threshold=0.0)
    
    # Run comprehensive validation
    success = validator.run_comprehensive_topk_validation()
    
    print("\n" + "="*80)
    if success:
        print("🎉 PHASE 4B.1 VALIDATION COMPLETE - ALL TESTS PASSED")
        print("✅ Top-K filtering alignment achieved with identical token selection")
    else:
        print("❌ PHASE 4B.1 VALIDATION FAILED")
        print("🔧 Top-K filtering requires further optimization")
    print("="*80)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 