#!/usr/bin/env python3
"""
Phase 4B.2: Top-P (Nucleus) Sampling Precision Validation
Building on Phase 3 methodology to achieve bit-exact nucleus sampling behavior
Target: Perfect cumulative probability computation and threshold cutoffs
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
logger = logging.getLogger("phase4b2_topp")

class Phase4B2TopPValidator:
    """Phase 4B.2: Top-P nucleus sampling precision validation using Phase 3 methodology"""
    
    def __init__(self, precision_threshold: float = 1e-7):
        """Initialize with Phase 3 precision standards"""
        self.precision_threshold = precision_threshold
        self.test_p_values = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
        self.edge_p_values = [0.0, 0.001, 0.999, 1.0]  # Edge cases
        self.vocab_sizes = [100, 1000, 10000, 32000, 152064]  # Including Qwen vocab size
        self.validation_passed = True
        
    def log_validation_result(self, test_name: str, passed: bool, details: str = ""):
        """Phase 3 style validation logging"""
        status = "✅" if passed else "❌"
        logger.info(f"{status} {test_name}: {details}")
        if not passed:
            self.validation_passed = False
    
    def create_test_logits(self, vocab_size: int, seed: int = 42, distribution_type: str = "normal") -> np.ndarray:
        """Create reproducible test logits with different distributions"""
        np.random.seed(seed)
        
        if distribution_type == "normal":
            logits = np.random.randn(vocab_size).astype(np.float32)
        elif distribution_type == "uniform":
            logits = np.random.uniform(-5, 5, vocab_size).astype(np.float32)
        elif distribution_type == "peaked":
            # Create a distribution with one very high peak
            logits = np.random.randn(vocab_size).astype(np.float32) * 0.1
            logits[0] = 10.0  # Very high peak
        elif distribution_type == "flat":
            # Nearly uniform distribution
            logits = np.random.randn(vocab_size).astype(np.float32) * 0.01
        else:
            logits = np.random.randn(vocab_size).astype(np.float32)
        
        return logits
    
    def apply_jax_topp_filtering(self, logits: np.ndarray, p: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply top-p filtering using JAX implementation"""
        jax_logits = jnp.array(logits)
        
        if p >= 1.0:
            # Keep all tokens
            return np.array(jax_logits), np.arange(len(logits)), np.ones_like(logits, dtype=bool)
        
        if p <= 0.0:
            # Keep only the highest token
            argmax_idx = jnp.argmax(jax_logits)
            mask = jnp.zeros_like(jax_logits, dtype=bool)
            mask = mask.at[argmax_idx].set(True)
            filtered_logits = jnp.where(mask, jax_logits, -jnp.inf)
            return np.array(filtered_logits), np.array([argmax_idx]), np.array(mask)
        
        # Sort logits in descending order
        sorted_indices = jnp.argsort(jax_logits)[::-1]
        sorted_logits = jax_logits[sorted_indices]
        
        # Convert to probabilities
        sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
        
        # Compute cumulative probabilities
        cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
        
        # Find indices to remove (cumulative prob > p)
        sorted_indices_to_remove = cumulative_probs > p
        
        # Keep at least the first token
        sorted_indices_to_remove = sorted_indices_to_remove.at[0].set(False)
        
        # Map back to original indices
        indices_to_remove = jnp.zeros_like(jax_logits, dtype=bool)
        indices_to_remove = indices_to_remove.at[sorted_indices].set(sorted_indices_to_remove)
        
        # Create filtered logits
        filtered_logits = jnp.where(~indices_to_remove, jax_logits, -jnp.inf)
        
        # Get kept indices
        kept_indices = jnp.where(~indices_to_remove)[0]
        
        return np.array(filtered_logits), np.array(kept_indices), np.array(~indices_to_remove)
    
    def apply_torch_topp_filtering(self, logits: np.ndarray, p: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply top-p filtering using PyTorch implementation"""
        torch_logits = torch.tensor(logits)
        
        if p >= 1.0:
            # Keep all tokens
            return torch_logits.numpy(), np.arange(len(logits)), np.ones_like(logits, dtype=bool)
        
        if p <= 0.0:
            # Keep only the highest token
            argmax_idx = torch.argmax(torch_logits)
            mask = torch.zeros_like(torch_logits, dtype=torch.bool)
            mask[argmax_idx] = True
            filtered_logits = torch.where(mask, torch_logits, torch.tensor(-float('inf')))
            return filtered_logits.numpy(), np.array([argmax_idx.item()]), mask.numpy()
        
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(torch_logits, descending=True)
        
        # Convert to probabilities
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        
        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find indices to remove (cumulative prob > p)
        sorted_indices_to_remove = cumulative_probs > p
        
        # Keep at least the first token
        sorted_indices_to_remove[0] = False
        
        # Map back to original indices
        indices_to_remove = torch.zeros_like(torch_logits, dtype=torch.bool)
        indices_to_remove[sorted_indices] = sorted_indices_to_remove
        
        # Create filtered logits
        filtered_logits = torch.where(~indices_to_remove, torch_logits, torch.tensor(-float('inf')))
        
        # Get kept indices
        kept_indices = torch.where(~indices_to_remove)[0]
        
        return filtered_logits.numpy(), kept_indices.numpy(), (~indices_to_remove).numpy()
    
    def validate_cumsum_precision(self, vocab_size: int, distribution_type: str = "normal") -> bool:
        """Validate cumulative sum precision between frameworks"""
        
        test_logits = self.create_test_logits(vocab_size, distribution_type=distribution_type)
        
        # JAX cumulative sum computation
        jax_logits = jnp.array(test_logits)
        jax_sorted_indices = jnp.argsort(jax_logits)[::-1]
        jax_sorted_logits = jax_logits[jax_sorted_indices]
        jax_probs = jax.nn.softmax(jax_sorted_logits, axis=-1)
        jax_cumsum = jnp.cumsum(jax_probs, axis=-1)
        
        # PyTorch cumulative sum computation
        torch_logits = torch.tensor(test_logits)
        torch_sorted_logits, torch_sorted_indices = torch.sort(torch_logits, descending=True)
        torch_probs = torch.softmax(torch_sorted_logits, dim=-1)
        torch_cumsum = torch.cumsum(torch_probs, dim=-1)
        
        # Compare cumulative sums
        jax_cumsum_np = np.array(jax_cumsum)
        torch_cumsum_np = torch_cumsum.numpy()
        
        max_diff = np.max(np.abs(jax_cumsum_np - torch_cumsum_np))
        mean_diff = np.mean(np.abs(jax_cumsum_np - torch_cumsum_np))
        
        precision_ok = max_diff < self.precision_threshold
        
        # Check that both reach 1.0 at the end (within tolerance)
        jax_final = float(jax_cumsum_np[-1])
        torch_final = float(torch_cumsum_np[-1])
        final_sum_ok = abs(jax_final - 1.0) < 1e-6 and abs(torch_final - 1.0) < 1e-6
        
        self.log_validation_result(
            f"Cumsum precision vocab={vocab_size}, dist={distribution_type}",
            precision_ok and final_sum_ok,
            f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, jax_final={jax_final:.6f}, torch_final={torch_final:.6f}"
        )
        
        return precision_ok and final_sum_ok
    
    def validate_topp_filtering_precision(self, vocab_size: int, p: float, distribution_type: str = "normal") -> bool:
        """Validate top-p filtering precision with Phase 3 standards"""
        
        test_logits = self.create_test_logits(vocab_size, distribution_type=distribution_type)
        
        # Apply top-p filtering
        jax_filtered, jax_indices, jax_mask = self.apply_jax_topp_filtering(test_logits, p)
        torch_filtered, torch_indices, torch_mask = self.apply_torch_topp_filtering(test_logits, p)
        
        # Compare masks (most important for top-p)
        masks_match = np.array_equal(jax_mask, torch_mask)
        
        # Compare kept indices (should be identical sets)
        jax_indices_sorted = np.sort(jax_indices)
        torch_indices_sorted = np.sort(torch_indices)
        indices_match = np.array_equal(jax_indices_sorted, torch_indices_sorted)
        
        # Compare finite values in filtered logits
        jax_finite_mask = np.isfinite(jax_filtered)
        torch_finite_mask = np.isfinite(torch_filtered)
        finite_masks_match = np.array_equal(jax_finite_mask, torch_finite_mask)
        
        precision_ok = True
        max_diff = 0.0
        
        if finite_masks_match and masks_match:
            jax_finite_values = jax_filtered[jax_finite_mask]
            torch_finite_values = torch_filtered[torch_finite_mask]
            
            if len(jax_finite_values) > 0 and len(torch_finite_values) > 0:
                max_diff = np.max(np.abs(jax_finite_values - torch_finite_values))
                precision_ok = max_diff < self.precision_threshold
        else:
            precision_ok = False
            max_diff = float('inf')
        
        all_match = masks_match and indices_match and finite_masks_match and precision_ok
        
        self.log_validation_result(
            f"Top-p filtering p={p:.3f}, vocab={vocab_size}, dist={distribution_type}",
            all_match,
            f"masks_match={masks_match}, indices_match={indices_match}, max_diff={max_diff:.2e}"
        )
        
        return all_match
    
    def validate_topp_probability_conservation(self, vocab_size: int, p: float) -> bool:
        """Validate that probability mass is conserved after top-p filtering"""
        
        test_logits = self.create_test_logits(vocab_size)
        
        # Apply top-p filtering
        jax_filtered, _, _ = self.apply_jax_topp_filtering(test_logits, p)
        torch_filtered, _, _ = self.apply_torch_topp_filtering(test_logits, p)
        
        # Convert to probabilities
        jax_probs = jax.nn.softmax(jnp.array(jax_filtered), axis=-1)
        torch_probs = torch.softmax(torch.tensor(torch_filtered), dim=-1)
        
        # Check probability sums
        jax_sum = float(jnp.sum(jax_probs))
        torch_sum = float(torch.sum(torch_probs))
        
        # Both should sum to 1.0
        jax_conservation_ok = abs(jax_sum - 1.0) < 1e-6
        torch_conservation_ok = abs(torch_sum - 1.0) < 1e-6
        
        # Check that both frameworks give similar probability mass
        sum_diff = abs(jax_sum - torch_sum)
        sum_precision_ok = sum_diff < self.precision_threshold
        
        conservation_ok = jax_conservation_ok and torch_conservation_ok and sum_precision_ok
        
        self.log_validation_result(
            f"Top-p probability conservation p={p:.3f}, vocab={vocab_size}",
            conservation_ok,
            f"jax_sum={jax_sum:.6f}, torch_sum={torch_sum:.6f}, sum_diff={sum_diff:.2e}"
        )
        
        return conservation_ok
    
    def test_topp_edge_cases(self, vocab_size: int) -> bool:
        """Test top-p edge cases using Phase 3 methodology"""
        
        edge_cases_passed = True
        test_logits = self.create_test_logits(vocab_size)
        
        # Edge case 1: p=0 (should keep only argmax)
        jax_filtered_0, jax_indices_0, _ = self.apply_jax_topp_filtering(test_logits, 0.0)
        torch_filtered_0, torch_indices_0, _ = self.apply_torch_topp_filtering(test_logits, 0.0)
        
        true_argmax = np.argmax(test_logits)
        case1_ok = (len(jax_indices_0) == 1 and len(torch_indices_0) == 1 and
                   jax_indices_0[0] == true_argmax and torch_indices_0[0] == true_argmax)
        
        if not case1_ok:
            edge_cases_passed = False
            self.log_validation_result(
                f"Top-p edge case p=0.0, vocab={vocab_size}", 
                False, 
                f"jax_indices={jax_indices_0}, torch_indices={torch_indices_0}, true_argmax={true_argmax}"
            )
        
        # Edge case 2: p=1.0 (should keep all tokens)
        jax_filtered_1, jax_indices_1, _ = self.apply_jax_topp_filtering(test_logits, 1.0)
        torch_filtered_1, torch_indices_1, _ = self.apply_torch_topp_filtering(test_logits, 1.0)
        
        case2_ok = (len(jax_indices_1) == vocab_size and len(torch_indices_1) == vocab_size and
                   np.allclose(jax_filtered_1, test_logits, atol=self.precision_threshold) and
                   np.allclose(torch_filtered_1, test_logits, atol=self.precision_threshold))
        
        if not case2_ok:
            edge_cases_passed = False
            self.log_validation_result(
                f"Top-p edge case p=1.0, vocab={vocab_size}", 
                False, 
                f"jax_kept={len(jax_indices_1)}, torch_kept={len(torch_indices_1)}, vocab_size={vocab_size}"
            )
        
        # Edge case 3: Very small p (should keep very few tokens)
        small_p = 0.001
        jax_filtered_small, jax_indices_small, _ = self.apply_jax_topp_filtering(test_logits, small_p)
        torch_filtered_small, torch_indices_small, _ = self.apply_torch_topp_filtering(test_logits, small_p)
        
        # Should keep at least 1 token (the argmax) but very few
        case3_ok = (len(jax_indices_small) >= 1 and len(torch_indices_small) >= 1 and
                   len(jax_indices_small) < vocab_size // 10 and len(torch_indices_small) < vocab_size // 10 and
                   len(jax_indices_small) == len(torch_indices_small))
        
        if not case3_ok:
            edge_cases_passed = False
            self.log_validation_result(
                f"Top-p edge case p={small_p}, vocab={vocab_size}", 
                False, 
                f"jax_kept={len(jax_indices_small)}, torch_kept={len(torch_indices_small)}"
            )
        
        if edge_cases_passed:
            self.log_validation_result(f"Top-p edge cases vocab={vocab_size}", True, "All edge cases passed")
        
        return edge_cases_passed
    
    def validate_topp_monotonicity(self, vocab_size: int) -> bool:
        """Validate that increasing p monotonically increases kept tokens"""
        
        test_logits = self.create_test_logits(vocab_size)
        p_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        jax_kept_counts = []
        torch_kept_counts = []
        
        for p in p_values:
            _, jax_indices, _ = self.apply_jax_topp_filtering(test_logits, p)
            _, torch_indices, _ = self.apply_torch_topp_filtering(test_logits, p)
            
            jax_kept_counts.append(len(jax_indices))
            torch_kept_counts.append(len(torch_indices))
        
        # Check monotonicity for JAX
        jax_monotonic = all(jax_kept_counts[i] <= jax_kept_counts[i+1] for i in range(len(jax_kept_counts)-1))
        
        # Check monotonicity for PyTorch
        torch_monotonic = all(torch_kept_counts[i] <= torch_kept_counts[i+1] for i in range(len(torch_kept_counts)-1))
        
        # Check that both frameworks give same counts
        counts_match = jax_kept_counts == torch_kept_counts
        
        monotonicity_ok = jax_monotonic and torch_monotonic and counts_match
        
        self.log_validation_result(
            f"Top-p monotonicity vocab={vocab_size}",
            monotonicity_ok,
            f"jax_counts={jax_kept_counts}, torch_counts={torch_kept_counts}, "
            f"jax_monotonic={jax_monotonic}, torch_monotonic={torch_monotonic}"
        )
        
        return monotonicity_ok
    
    def validate_topp_different_distributions(self, vocab_size: int, p: float) -> bool:
        """Validate top-p behavior across different probability distributions"""
        
        distributions = ["normal", "uniform", "peaked", "flat"]
        all_distributions_ok = True
        
        for dist_type in distributions:
            test_logits = self.create_test_logits(vocab_size, distribution_type=dist_type)
            
            # Apply top-p filtering
            jax_filtered, jax_indices, jax_mask = self.apply_jax_topp_filtering(test_logits, p)
            torch_filtered, torch_indices, torch_mask = self.apply_torch_topp_filtering(test_logits, p)
            
            # Check consistency
            indices_match = np.array_equal(np.sort(jax_indices), np.sort(torch_indices))
            masks_match = np.array_equal(jax_mask, torch_mask)
            
            dist_ok = indices_match and masks_match
            all_distributions_ok = all_distributions_ok and dist_ok
            
            if not dist_ok:
                self.log_validation_result(
                    f"Top-p distribution {dist_type} p={p:.3f}, vocab={vocab_size}",
                    False,
                    f"indices_match={indices_match}, masks_match={masks_match}"
                )
        
        if all_distributions_ok:
            self.log_validation_result(
                f"Top-p different distributions p={p:.3f}, vocab={vocab_size}",
                True,
                "All distribution types consistent"
            )
        
        return all_distributions_ok
    
    def run_comprehensive_topp_validation(self) -> bool:
        """Run comprehensive top-p validation using Phase 3 methodology"""
        
        logger.info("🚀 Starting Phase 4B.2: Top-P Nucleus Sampling Precision Validation")
        logger.info(f"Target precision: {self.precision_threshold:.2e}")
        
        # Test 1: Cumulative sum precision
        logger.info("📊 Testing cumulative sum precision...")
        for vocab_size in [1000, 10000, 32000]:
            for dist_type in ["normal", "peaked", "flat"]:
                self.validate_cumsum_precision(vocab_size, dist_type)
        
        # Test 2: Top-p filtering precision
        logger.info("📊 Testing top-p filtering precision...")
        for vocab_size in self.vocab_sizes:
            for p in self.test_p_values:
                if vocab_size <= 10000:  # Test all distributions for smaller vocabs
                    self.validate_topp_filtering_precision(vocab_size, p)
                else:  # Test only normal distribution for large vocabs
                    self.validate_topp_filtering_precision(vocab_size, p, "normal")
        
        # Test 3: Probability conservation
        logger.info("📊 Testing probability conservation...")
        for vocab_size in [1000, 10000]:
            for p in [0.1, 0.5, 0.8, 0.95]:
                self.validate_topp_probability_conservation(vocab_size, p)
        
        # Test 4: Edge cases
        logger.info("📊 Testing top-p edge cases...")
        for vocab_size in [100, 1000, 10000]:
            self.test_topp_edge_cases(vocab_size)
        
        # Test 5: Monotonicity
        logger.info("📊 Testing top-p monotonicity...")
        for vocab_size in [100, 1000]:
            self.validate_topp_monotonicity(vocab_size)
        
        # Test 6: Different distributions
        logger.info("📊 Testing top-p across different distributions...")
        for vocab_size in [100, 1000]:
            for p in [0.3, 0.7, 0.9]:
                self.validate_topp_different_distributions(vocab_size, p)
        
        # Final validation summary
        if self.validation_passed:
            logger.info("🎉 Phase 4B.2 PASSED: Top-P nucleus sampling validation successful!")
            logger.info("✅ Achieved bit-exact nucleus sampling behavior")
        else:
            logger.error("❌ Phase 4B.2 FAILED: Top-P nucleus sampling alignment issues detected")
            logger.error("🔧 Review top-p implementation for precision improvements")
        
        return self.validation_passed

def main():
    """Main Phase 4B.2 validation entry point"""
    
    print("="*80)
    print("PHASE 4B.2: TOP-P NUCLEUS SAMPLING PRECISION VALIDATION")
    print("Building on Phase 3 methodology for bit-exact nucleus sampling behavior")
    print("="*80)
    
    # Initialize validator with Phase 3 precision standards
    validator = Phase4B2TopPValidator(precision_threshold=1e-7)
    
    # Run comprehensive validation
    success = validator.run_comprehensive_topp_validation()
    
    print("\n" + "="*80)
    if success:
        print("🎉 PHASE 4B.2 VALIDATION COMPLETE - ALL TESTS PASSED")
        print("✅ Top-P nucleus sampling precision alignment achieved")
    else:
        print("❌ PHASE 4B.2 VALIDATION FAILED")
        print("🔧 Top-P nucleus sampling requires further optimization")
    print("="*80)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 