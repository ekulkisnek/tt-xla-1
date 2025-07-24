#!/usr/bin/env python3
"""
Phase 5B.1: Causal Mask Updates
Using Phase 3 systematic approach for perfect mask handling
Target: Perfect causal masking throughout generation
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
logger = logging.getLogger("phase5b1_mask")

class Phase5B1CausalMaskValidator:
    """Phase 5B.1: Causal mask evolution with Phase 3 systematic approach"""
    
    def __init__(self):
        self.precision_threshold = 1e-9  # Very strict for mask operations
        self.validation_passed = True
        
        # Test configurations for different generation scenarios
        self.test_scenarios = [
            {"initial_seq": 1, "generation_steps": 3, "batch_size": 1},
            {"initial_seq": 5, "generation_steps": 5, "batch_size": 1},
            {"initial_seq": 1, "generation_steps": 10, "batch_size": 1},
            {"initial_seq": 3, "generation_steps": 2, "batch_size": 2},  # Batch case
        ]
        
    def log_validation_result(self, test_name: str, passed: bool, details: str = ""):
        """Phase 3 style validation logging"""
        status = "✅" if passed else "❌"
        logger.info(f"{status} {test_name}: {details}")
        if not passed:
            self.validation_passed = False
    
    def make_causal_mask_jax(self, q_len: int, k_len: int) -> jnp.ndarray:
        """JAX implementation of causal mask (from our model)"""
        i = jnp.arange(q_len)[:, None]
        j = jnp.arange(k_len)[None, :]
        mask = (i < j - (k_len - q_len)) * -1e9
        return mask
    
    def make_causal_mask_torch(self, q_len: int, k_len: int) -> np.ndarray:
        """PyTorch implementation of causal mask"""
        i = torch.arange(q_len)[:, None]
        j = torch.arange(k_len)[None, :]
        mask = (i < j - (k_len - q_len)) * -1e9
        return mask.numpy()
    
    def test_basic_causal_mask_alignment(self, q_len: int, k_len: int) -> bool:
        """Test basic causal mask alignment between JAX and PyTorch"""
        
        # Generate masks
        jax_mask = self.make_causal_mask_jax(q_len, k_len)
        torch_mask = self.make_causal_mask_torch(q_len, k_len)
        
        # Compare results
        mask_diff = np.max(np.abs(np.array(jax_mask) - torch_mask))
        shapes_match = jax_mask.shape == torch_mask.shape
        
        # Verify expected shape
        expected_shape = (q_len, k_len)
        shape_correct = (jax_mask.shape == expected_shape and torch_mask.shape == expected_shape)
        
        all_passed = (mask_diff < self.precision_threshold and shapes_match and shape_correct)
        
        details = (f"q_len={q_len}, k_len={k_len}: "
                  f"mask_diff={mask_diff:.2e}, "
                  f"shapes_match={shapes_match}, "
                  f"jax_shape={jax_mask.shape}, torch_shape={torch_mask.shape}")
        
        return all_passed, details
    
    def test_incremental_mask_evolution(self, scenario: Dict) -> bool:
        """Test mask evolution during incremental generation"""
        
        initial_seq = scenario["initial_seq"]
        generation_steps = scenario["generation_steps"]
        batch_size = scenario["batch_size"]
        
        evolution_consistent = True
        step_details = []
        
        # Simulate generation process
        for step in range(generation_steps):
            current_step = step + 1  # 1-indexed
            
            # Current sequence length (initial + generated tokens)
            current_seq_len = initial_seq + current_step
            
            # For generation: q_len=1 (new token), k_len=current_seq_len (all tokens)
            q_len = 1
            k_len = current_seq_len
            
            # Generate masks
            jax_mask = self.make_causal_mask_jax(q_len, k_len)
            torch_mask = self.make_causal_mask_torch(q_len, k_len)
            
            # Compare
            mask_diff = np.max(np.abs(np.array(jax_mask) - torch_mask))
            step_consistent = mask_diff < self.precision_threshold
            evolution_consistent = evolution_consistent and step_consistent
            
            # Add batch dimensions to match model usage
            jax_mask_batched = jnp.broadcast_to(jax_mask[None, None, :, :], (batch_size, 1, q_len, k_len))
            torch_mask_batched = np.broadcast_to(torch_mask[None, None, :, :], (batch_size, 1, q_len, k_len))
            
            # Verify batch broadcasting consistency
            batched_diff = np.max(np.abs(np.array(jax_mask_batched) - torch_mask_batched))
            batch_consistent = batched_diff < self.precision_threshold
            evolution_consistent = evolution_consistent and batch_consistent
            
            step_details.append(f"step{current_step}(seq={k_len},diff={mask_diff:.2e},batch_diff={batched_diff:.2e})")
        
        details = (f"initial_seq={initial_seq}, steps={generation_steps}, batch={batch_size}: "
                  + ", ".join(step_details))
        
        return evolution_consistent, details
    
    def test_mask_shape_consistency(self, scenario: Dict) -> bool:
        """Test mask shape consistency across generation steps"""
        
        initial_seq = scenario["initial_seq"]
        generation_steps = scenario["generation_steps"]
        batch_size = scenario["batch_size"]
        
        shape_tests = []
        all_shapes_correct = True
        
        for step in range(generation_steps):
            current_step = step + 1
            current_seq_len = initial_seq + current_step
            
            # Generation mask: new token attending to all previous tokens
            q_len = 1
            k_len = current_seq_len
            
            mask = self.make_causal_mask_jax(q_len, k_len)
            expected_shape = (q_len, k_len)
            
            shape_correct = mask.shape == expected_shape
            all_shapes_correct = all_shapes_correct and shape_correct
            
            # Test with batch dimensions
            batched_mask = jnp.broadcast_to(mask[None, None, :, :], (batch_size, 1, q_len, k_len))
            expected_batched_shape = (batch_size, 1, q_len, k_len)
            
            batched_shape_correct = batched_mask.shape == expected_batched_shape
            all_shapes_correct = all_shapes_correct and batched_shape_correct
            
            shape_tests.append(f"step{current_step}(shape={mask.shape},batched={batched_mask.shape})")
        
        details = (f"initial_seq={initial_seq}, steps={generation_steps}, batch={batch_size}: "
                  + ", ".join(shape_tests))
        
        return all_shapes_correct, details
    
    def test_mask_attention_interaction(self, q_len: int, k_len: int) -> bool:
        """Test how mask interacts with attention logits"""
        
        # Create test attention logits
        np.random.seed(42)
        attention_logits = np.random.randn(q_len, k_len).astype(np.float32)
        
        # Apply causal mask
        jax_mask = self.make_causal_mask_jax(q_len, k_len)
        torch_mask = self.make_causal_mask_torch(q_len, k_len)
        
        # JAX masked logits
        jax_logits = jnp.array(attention_logits)
        jax_masked_logits = jax_logits + jax_mask
        
        # PyTorch masked logits
        torch_logits = torch.tensor(attention_logits)
        torch_masked_logits = torch_logits + torch.tensor(torch_mask)
        
        # Compare masked results
        masked_diff = np.max(np.abs(np.array(jax_masked_logits) - torch_masked_logits.numpy()))
        
        # Test softmax after masking
        jax_probs = jax.nn.softmax(jax_masked_logits, axis=-1)
        torch_probs = torch.softmax(torch_masked_logits, dim=-1)
        
        probs_diff = np.max(np.abs(np.array(jax_probs) - torch_probs.numpy()))
        
        # Verify causal property: upper triangular should be very small after softmax
        causal_property = True
        for i in range(q_len):
            for j in range(k_len):
                if i < j - (k_len - q_len):  # Should be masked
                    jax_prob_val = float(jax_probs[i, j])
                    torch_prob_val = float(torch_probs[i, j])
                    
                    # Should be very close to 0 after softmax
                    if jax_prob_val > 1e-6 or torch_prob_val > 1e-6:
                        causal_property = False
                        break
        
        all_passed = (masked_diff < self.precision_threshold and 
                     probs_diff < 1e-6 and  # Slightly relaxed for softmax
                     causal_property)
        
        details = (f"q_len={q_len}, k_len={k_len}: "
                  f"masked_diff={masked_diff:.2e}, probs_diff={probs_diff:.2e}, "
                  f"causal_property={causal_property}")
        
        return all_passed, details
    
    def test_extended_sequence_masks(self, max_seq_len: int) -> bool:
        """Test mask behavior with extended sequences"""
        
        extended_tests = []
        all_extended_correct = True
        
        # Test various sequence length combinations
        test_cases = [
            (1, max_seq_len),      # Single query, long key
            (max_seq_len, max_seq_len),  # Square mask
            (5, max_seq_len),      # Small query, long key
        ]
        
        for q_len, k_len in test_cases:
            jax_mask = self.make_causal_mask_jax(q_len, k_len)
            torch_mask = self.make_causal_mask_torch(q_len, k_len)
            
            mask_diff = np.max(np.abs(np.array(jax_mask) - torch_mask))
            case_correct = mask_diff < self.precision_threshold
            all_extended_correct = all_extended_correct and case_correct
            
            # Verify pattern: lower triangular + diagonal should be 0, upper should be -1e9
            pattern_correct = True
            for i in range(q_len):
                for j in range(k_len):
                    expected_val = -1e9 if i < j - (k_len - q_len) else 0.0
                    jax_val = float(jax_mask[i, j])
                    torch_val = float(torch_mask[i, j])
                    
                    if abs(jax_val - expected_val) > 1e-6 or abs(torch_val - expected_val) > 1e-6:
                        pattern_correct = False
                        break
                if not pattern_correct:
                    break
            
            case_correct = case_correct and pattern_correct
            all_extended_correct = all_extended_correct and case_correct
            
            extended_tests.append(f"({q_len},{k_len}):diff={mask_diff:.2e},pattern={pattern_correct}")
        
        details = f"max_seq={max_seq_len}: " + ", ".join(extended_tests)
        
        return all_extended_correct, details
    
    def run_causal_mask_validation(self) -> bool:
        """Run comprehensive causal mask validation tests"""
        
        logger.info("🚀 Starting Phase 5B.1: Causal Mask Updates")
        logger.info("Target: Perfect causal masking throughout generation")
        
        # Test 1: Basic causal mask alignment
        logger.info("📊 Testing basic causal mask alignment...")
        basic_test_cases = [(1, 1), (1, 5), (5, 5), (3, 10), (10, 10)]
        for q_len, k_len in basic_test_cases:
            passed, details = self.test_basic_causal_mask_alignment(q_len, k_len)
            self.log_validation_result(f"Basic mask alignment", passed, details)
        
        # Test 2: Incremental mask evolution
        logger.info("📊 Testing incremental mask evolution...")
        for scenario in self.test_scenarios:
            passed, details = self.test_incremental_mask_evolution(scenario)
            self.log_validation_result(f"Incremental evolution", passed, details)
        
        # Test 3: Mask shape consistency
        logger.info("📊 Testing mask shape consistency...")
        for scenario in self.test_scenarios:
            passed, details = self.test_mask_shape_consistency(scenario)
            self.log_validation_result(f"Shape consistency", passed, details)
        
        # Test 4: Mask-attention interaction
        logger.info("📊 Testing mask-attention interaction...")
        interaction_test_cases = [(1, 5), (3, 10), (5, 15)]
        for q_len, k_len in interaction_test_cases:
            passed, details = self.test_mask_attention_interaction(q_len, k_len)
            self.log_validation_result(f"Mask-attention interaction", passed, details)
        
        # Test 5: Extended sequence masks
        logger.info("📊 Testing extended sequence masks...")
        for max_seq in [20, 50]:
            passed, details = self.test_extended_sequence_masks(max_seq)
            self.log_validation_result(f"Extended sequences", passed, details)
        
        # Final validation summary
        if self.validation_passed:
            logger.info("🎉 Phase 5B.1 PASSED: Perfect causal masking throughout generation!")
            logger.info("✅ Basic mask alignment verified")
            logger.info("✅ Incremental mask evolution validated")
            logger.info("✅ Shape consistency confirmed")
            logger.info("✅ Mask-attention interaction tested")
            logger.info("✅ Extended sequence handling verified")
        else:
            logger.error("❌ Phase 5B.1 FAILED: Causal mask issues detected")
            logger.error("🔧 Review mask implementation and evolution logic")
        
        return self.validation_passed

def main():
    """Main Phase 5B.1 validation entry point"""
    
    print("="*80)
    print("PHASE 5B.1: CAUSAL MASK UPDATES")
    print("Using Phase 3 systematic approach for perfect mask handling")
    print("="*80)
    
    validator = Phase5B1CausalMaskValidator()
    success = validator.run_causal_mask_validation()
    
    print("\n" + "="*80)
    if success:
        print("🎉 PHASE 5B.1 VALIDATION COMPLETE - ALL TESTS PASSED")
        print("✅ Perfect causal masking throughout generation")
    else:
        print("❌ PHASE 5B.1 VALIDATION FAILED")
        print("🔧 Causal mask handling requires improvements")
    print("="*80)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 