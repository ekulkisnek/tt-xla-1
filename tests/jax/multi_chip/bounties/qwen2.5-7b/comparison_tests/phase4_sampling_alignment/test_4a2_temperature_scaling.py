#!/usr/bin/env python3
"""
Phase 4A.2: Temperature Scaling Precision Validation
Building on Phase 3 methodology to achieve perfect temperature scaling alignment
Target: Perfect temperature scaling alignment with numerical stability
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
logger = logging.getLogger("phase4a2_temperature")

class Phase4A2TemperatureValidator:
    """Phase 4A.2: Temperature scaling precision validation using Phase 3 methodology"""
    
    def __init__(self, precision_threshold: float = 1e-7):
        """Initialize with Phase 3 precision standards"""
        self.precision_threshold = precision_threshold
        self.test_temperatures = [0.01, 0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0]
        self.edge_temperatures = [1e-8, 1e-6, 1e-4, 100.0, 1000.0]  # Edge cases
        self.vocab_sizes = [100, 1000, 32000, 152064]  # Including Qwen vocab size
        self.validation_passed = True
        
    def log_validation_result(self, test_name: str, passed: bool, details: str = ""):
        """Phase 3 style validation logging"""
        status = "✅" if passed else "❌"
        logger.info(f"{status} {test_name}: {details}")
        if not passed:
            self.validation_passed = False
    
    def create_test_logits(self, vocab_size: int, seed: int = 42) -> np.ndarray:
        """Create reproducible test logits for validation"""
        np.random.seed(seed)
        # Create diverse logits with various scales
        logits = np.random.randn(vocab_size).astype(np.float32)
        
        # Add some extreme values to test numerical stability
        if vocab_size > 10:
            logits[0] = 20.0  # Very high logit
            logits[1] = -20.0  # Very low logit
            logits[2] = 0.0   # Neutral logit
        
        return logits
    
    def apply_jax_temperature_scaling(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling using JAX implementation"""
        jax_logits = jnp.array(logits)
        
        if temperature < 1e-5:
            # Handle near-zero temperature case
            return np.array(jax_logits)  # No scaling for greedy case
        
        scaled_logits = jax_logits / temperature
        return np.array(scaled_logits)
    
    def apply_torch_temperature_scaling(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling using PyTorch implementation"""
        torch_logits = torch.tensor(logits)
        
        if temperature < 1e-5:
            # Handle near-zero temperature case
            return torch_logits.numpy()  # No scaling for greedy case
        
        scaled_logits = torch_logits / temperature
        return scaled_logits.numpy()
    
    def validate_temperature_scaling_precision(self, vocab_size: int, temperature: float) -> bool:
        """Validate temperature scaling precision between frameworks"""
        
        test_logits = self.create_test_logits(vocab_size)
        
        # Apply temperature scaling
        jax_scaled = self.apply_jax_temperature_scaling(test_logits, temperature)
        torch_scaled = self.apply_torch_temperature_scaling(test_logits, temperature)
        
        # Check precision alignment
        max_diff = np.max(np.abs(jax_scaled - torch_scaled))
        mean_diff = np.mean(np.abs(jax_scaled - torch_scaled))
        
        precision_ok = max_diff < self.precision_threshold
        
        self.log_validation_result(
            f"Temperature scaling T={temperature:.6f}, vocab={vocab_size}",
            precision_ok,
            f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e} (threshold={self.precision_threshold:.2e})"
        )
        
        return precision_ok
    
    def validate_softmax_after_temperature(self, vocab_size: int, temperature: float) -> bool:
        """Validate softmax computation after temperature scaling"""
        
        test_logits = self.create_test_logits(vocab_size)
        
        # JAX: temperature scaling + softmax
        jax_logits = jnp.array(test_logits)
        if temperature >= 1e-5:
            jax_scaled = jax_logits / temperature
        else:
            jax_scaled = jax_logits
        jax_probs = jax.nn.softmax(jax_scaled, axis=-1)
        
        # PyTorch: temperature scaling + softmax
        torch_logits = torch.tensor(test_logits)
        if temperature >= 1e-5:
            torch_scaled = torch_logits / temperature
        else:
            torch_scaled = torch_logits
        torch_probs = torch.softmax(torch_scaled, dim=-1)
        
        # Compare probabilities
        jax_probs_np = np.array(jax_probs)
        torch_probs_np = torch_probs.numpy()
        
        max_diff = np.max(np.abs(jax_probs_np - torch_probs_np))
        prob_sum_jax = np.sum(jax_probs_np)
        prob_sum_torch = np.sum(torch_probs_np)
        
        precision_ok = max_diff < self.precision_threshold
        sum_ok = abs(prob_sum_jax - 1.0) < 1e-6 and abs(prob_sum_torch - 1.0) < 1e-6
        
        self.log_validation_result(
            f"Softmax after temp T={temperature:.6f}, vocab={vocab_size}",
            precision_ok and sum_ok,
            f"max_diff={max_diff:.2e}, sum_jax={prob_sum_jax:.6f}, sum_torch={prob_sum_torch:.6f}"
        )
        
        return precision_ok and sum_ok
    
    def test_numerical_stability_edge_cases(self, temperature: float) -> bool:
        """Test numerical stability for edge cases"""
        
        # Create extreme logits for stability testing
        extreme_logits = np.array([
            100.0, -100.0, 0.0,  # Extreme values
            1e10, -1e10,          # Very extreme values
            np.inf, -np.inf,      # Infinite values
            np.nan                # NaN value
        ], dtype=np.float32)
        
        try:
            # JAX handling
            jax_logits = jnp.array(extreme_logits)
            if temperature >= 1e-5:
                jax_scaled = jax_logits / temperature
            else:
                jax_scaled = jax_logits
            
            # Check for NaN/Inf propagation
            jax_has_nan = jnp.any(jnp.isnan(jax_scaled))
            jax_has_inf = jnp.any(jnp.isinf(jax_scaled))
            
            # PyTorch handling
            torch_logits = torch.tensor(extreme_logits)
            if temperature >= 1e-5:
                torch_scaled = torch_logits / temperature
            else:
                torch_scaled = torch_logits
            
            torch_has_nan = torch.any(torch.isnan(torch_scaled))
            torch_has_inf = torch.any(torch.isinf(torch_scaled))
            
            # Both should handle edge cases consistently
            stability_ok = (jax_has_nan == torch_has_nan and jax_has_inf == torch_has_inf)
            
            self.log_validation_result(
                f"Stability edge cases T={temperature:.6f}",
                stability_ok,
                f"JAX(nan={jax_has_nan},inf={jax_has_inf}) PyTorch(nan={torch_has_nan},inf={torch_has_inf})"
            )
            
            return stability_ok
            
        except Exception as e:
            self.log_validation_result(
                f"Stability edge cases T={temperature:.6f}",
                False,
                f"Exception: {e}"
            )
            return False
    
    def validate_greedy_temperature_behavior(self, vocab_size: int) -> bool:
        """Validate behavior at very low temperatures (greedy decoding)"""
        
        test_logits = self.create_test_logits(vocab_size)
        low_temps = [0.0, 1e-8, 1e-6, 1e-4]
        
        # Find true argmax
        true_argmax = np.argmax(test_logits)
        
        all_consistent = True
        
        for temp in low_temps:
            # JAX processing
            jax_logits = jnp.array(test_logits)
            if temp < 1e-5:
                jax_scaled = jax_logits  # No scaling
                jax_argmax = jnp.argmax(jax_scaled)
            else:
                jax_scaled = jax_logits / temp
                jax_probs = jax.nn.softmax(jax_scaled, axis=-1)
                jax_argmax = jnp.argmax(jax_probs)
            
            # PyTorch processing
            torch_logits = torch.tensor(test_logits)
            if temp < 1e-5:
                torch_scaled = torch_logits  # No scaling
                torch_argmax = torch.argmax(torch_scaled)
            else:
                torch_scaled = torch_logits / temp
                torch_probs = torch.softmax(torch_scaled, dim=-1)
                torch_argmax = torch.argmax(torch_probs)
            
            # All should give same argmax
            consistent = (int(jax_argmax) == int(torch_argmax) == true_argmax)
            all_consistent = all_consistent and consistent
            
            if not consistent:
                self.log_validation_result(
                    f"Greedy consistency T={temp:.6f}, vocab={vocab_size}",
                    False,
                    f"jax_argmax={int(jax_argmax)}, torch_argmax={int(torch_argmax)}, true_argmax={true_argmax}"
                )
        
        if all_consistent:
            self.log_validation_result(
                f"Greedy temperature behavior vocab={vocab_size}",
                True,
                "All low temperatures produce consistent argmax"
            )
        
        return all_consistent
    
    def validate_temperature_monotonicity(self, vocab_size: int) -> bool:
        """Validate that temperature scaling behaves monotonically"""
        
        test_logits = self.create_test_logits(vocab_size)
        
        # Test temperature effect on probability distribution
        temp_low = 0.1
        temp_high = 2.0
        
        # JAX probabilities at different temperatures
        jax_logits = jnp.array(test_logits)
        jax_probs_low = jax.nn.softmax(jax_logits / temp_low, axis=-1)
        jax_probs_high = jax.nn.softmax(jax_logits / temp_high, axis=-1)
        
        # PyTorch probabilities at different temperatures
        torch_logits = torch.tensor(test_logits)
        torch_probs_low = torch.softmax(torch_logits / temp_low, dim=-1)
        torch_probs_high = torch.softmax(torch_logits / temp_high, dim=-1)
        
        # Low temperature should be more peaked (higher max probability)
        jax_max_low = float(jnp.max(jax_probs_low))
        jax_max_high = float(jnp.max(jax_probs_high))
        torch_max_low = float(torch.max(torch_probs_low))
        torch_max_high = float(torch.max(torch_probs_high))
        
        # Entropy should decrease with lower temperature
        jax_entropy_low = -float(jnp.sum(jax_probs_low * jnp.log(jax_probs_low + 1e-8)))
        jax_entropy_high = -float(jnp.sum(jax_probs_high * jnp.log(jax_probs_high + 1e-8)))
        torch_entropy_low = -float(torch.sum(torch_probs_low * torch.log(torch_probs_low + 1e-8)))
        torch_entropy_high = -float(torch.sum(torch_probs_high * torch.log(torch_probs_high + 1e-8)))
        
        monotonic_jax = (jax_max_low > jax_max_high and jax_entropy_low < jax_entropy_high)
        monotonic_torch = (torch_max_low > torch_max_high and torch_entropy_low < torch_entropy_high)
        
        consistent_monotonicity = monotonic_jax and monotonic_torch
        
        self.log_validation_result(
            f"Temperature monotonicity vocab={vocab_size}",
            consistent_monotonicity,
            f"JAX: max({jax_max_low:.3f}>{jax_max_high:.3f}) entropy({jax_entropy_low:.3f}<{jax_entropy_high:.3f}) | "
            f"PyTorch: max({torch_max_low:.3f}>{torch_max_high:.3f}) entropy({torch_entropy_low:.3f}<{torch_entropy_high:.3f})"
        )
        
        return consistent_monotonicity
    
    def run_comprehensive_temperature_validation(self) -> bool:
        """Run comprehensive temperature scaling validation using Phase 3 methodology"""
        
        logger.info("🚀 Starting Phase 4A.2: Temperature Scaling Precision Validation")
        logger.info(f"Target precision: {self.precision_threshold:.2e}")
        
        # Test 1: Basic temperature scaling precision
        logger.info("📊 Testing temperature scaling precision...")
        for vocab_size in self.vocab_sizes:
            for temperature in self.test_temperatures:
                self.validate_temperature_scaling_precision(vocab_size, temperature)
        
        # Test 2: Softmax after temperature scaling
        logger.info("📊 Testing softmax after temperature scaling...")
        for vocab_size in [1000, 32000]:  # Representative sizes
            for temperature in [0.1, 0.7, 1.0, 1.5]:
                self.validate_softmax_after_temperature(vocab_size, temperature)
        
        # Test 3: Edge case stability
        logger.info("📊 Testing numerical stability edge cases...")
        for temperature in self.edge_temperatures:
            self.test_numerical_stability_edge_cases(temperature)
        
        # Test 4: Greedy temperature behavior
        logger.info("📊 Testing greedy temperature behavior...")
        for vocab_size in [100, 1000, 32000]:
            self.validate_greedy_temperature_behavior(vocab_size)
        
        # Test 5: Temperature monotonicity
        logger.info("📊 Testing temperature monotonicity...")
        for vocab_size in [100, 1000]:
            self.validate_temperature_monotonicity(vocab_size)
        
        # Final validation summary
        if self.validation_passed:
            logger.info("🎉 Phase 4A.2 PASSED: Temperature scaling precision validation successful!")
            logger.info("✅ Achieved perfect temperature scaling alignment with numerical stability")
        else:
            logger.error("❌ Phase 4A.2 FAILED: Temperature scaling alignment issues detected")
            logger.error("🔧 Review temperature scaling implementation for precision improvements")
        
        return self.validation_passed

def main():
    """Main Phase 4A.2 validation entry point"""
    
    print("="*80)
    print("PHASE 4A.2: TEMPERATURE SCALING PRECISION VALIDATION")
    print("Building on Phase 3 methodology for perfect temperature scaling alignment")
    print("="*80)
    
    # Initialize validator with Phase 3 precision standards
    validator = Phase4A2TemperatureValidator(precision_threshold=1e-7)
    
    # Run comprehensive validation
    success = validator.run_comprehensive_temperature_validation()
    
    print("\n" + "="*80)
    if success:
        print("🎉 PHASE 4A.2 VALIDATION COMPLETE - ALL TESTS PASSED")
        print("✅ Temperature scaling precision alignment achieved with numerical stability")
    else:
        print("❌ PHASE 4A.2 VALIDATION FAILED")
        print("🔧 Temperature scaling requires further optimization")
    print("="*80)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 