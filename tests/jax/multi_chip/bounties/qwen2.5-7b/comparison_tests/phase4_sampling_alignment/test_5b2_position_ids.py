#!/usr/bin/env python3
"""
Phase 5B.2: Position ID Management
Building on Phase 3 RoPE expertise for perfect positional encoding alignment
Target: Perfect positional encoding alignment during generation
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
logger = logging.getLogger("phase5b2_position")

class Phase5B2PositionIDValidator:
    """Phase 5B.2: Position ID management with Phase 3 RoPE expertise"""
    
    def __init__(self):
        self.precision_threshold = 1e-9  # Very strict for position operations
        self.rope_precision_threshold = 1e-6  # More realistic for trigonometric operations (Phase 4 learning)
        self.validation_passed = True
        
        # Test configurations for position management
        self.test_scenarios = [
            {"initial_seq": 1, "generation_steps": 5, "batch_size": 1},
            {"initial_seq": 3, "generation_steps": 7, "batch_size": 1},
            {"initial_seq": 10, "generation_steps": 3, "batch_size": 1},
            {"initial_seq": 5, "generation_steps": 3, "batch_size": 2},  # Batch case
        ]
        
        # RoPE parameters (from Qwen model)
        self.head_dim = 128
        self.rope_theta = 10000.0
        
    def log_validation_result(self, test_name: str, passed: bool, details: str = ""):
        """Phase 3 style validation logging"""
        status = "✅" if passed else "❌"
        logger.info(f"{status} {test_name}: {details}")
        if not passed:
            self.validation_passed = False
    
    def create_position_ids_jax(self, batch_size: int, seq_len: int, offset: int = 0) -> jnp.ndarray:
        """JAX implementation of position ID creation"""
        position_ids = jnp.arange(offset, offset + seq_len, dtype=jnp.int32)[None, :]
        position_ids = jnp.broadcast_to(position_ids, (batch_size, seq_len))
        return position_ids
    
    def create_position_ids_torch(self, batch_size: int, seq_len: int, offset: int = 0) -> np.ndarray:
        """PyTorch implementation of position ID creation"""
        position_ids = torch.arange(offset, offset + seq_len, dtype=torch.int32)[None, :]
        position_ids = position_ids.expand(batch_size, seq_len)
        return position_ids.numpy()
    
    def compute_cos_sin_jax(self, position_ids: jnp.ndarray, head_dim: int, rope_theta: float = 10000.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """JAX RoPE cos/sin computation (from our model)"""
        pos = position_ids
        dim = head_dim // 2
        inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, dim, dtype=jnp.float32) / dim))
        freqs = jnp.einsum('bi,j->bij', pos.astype(jnp.float32), inv_freq)
        
        cos = jnp.cos(freqs)
        sin = jnp.sin(freqs)
        
        # Repeat to match head_dim
        cos = jnp.repeat(cos, 2, axis=-1)
        sin = jnp.repeat(sin, 2, axis=-1)
        
        return cos, sin
    
    def compute_cos_sin_torch(self, position_ids: np.ndarray, head_dim: int, rope_theta: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
        """PyTorch RoPE cos/sin computation"""
        pos = torch.tensor(position_ids, dtype=torch.float32)
        dim = head_dim // 2
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, dtype=torch.float32) / dim))
        freqs = torch.einsum('bi,j->bij', pos, inv_freq)
        
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        # Repeat to match head_dim
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
        
        return cos.numpy(), sin.numpy()
    
    def test_position_id_creation_alignment(self, batch_size: int, seq_len: int, offset: int = 0) -> bool:
        """Test position ID creation alignment between JAX and PyTorch"""
        
        # Create position IDs
        jax_positions = self.create_position_ids_jax(batch_size, seq_len, offset)
        torch_positions = self.create_position_ids_torch(batch_size, seq_len, offset)
        
        # Compare results
        pos_diff = np.max(np.abs(np.array(jax_positions) - torch_positions))
        shapes_match = jax_positions.shape == torch_positions.shape
        
        # Verify expected shape and values
        expected_shape = (batch_size, seq_len)
        shape_correct = (jax_positions.shape == expected_shape and torch_positions.shape == expected_shape)
        
        # Verify position values are correct
        expected_positions = np.arange(offset, offset + seq_len)[None, :]
        expected_positions = np.broadcast_to(expected_positions, (batch_size, seq_len))
        
        jax_values_correct = np.array_equal(np.array(jax_positions), expected_positions)
        torch_values_correct = np.array_equal(torch_positions, expected_positions)
        
        all_passed = (pos_diff == 0 and shapes_match and shape_correct and 
                     jax_values_correct and torch_values_correct)
        
        details = (f"batch={batch_size}, seq={seq_len}, offset={offset}: "
                  f"pos_diff={pos_diff}, shapes_match={shapes_match}, "
                  f"jax_values_correct={jax_values_correct}, torch_values_correct={torch_values_correct}")
        
        return all_passed, details
    
    def test_incremental_position_tracking(self, scenario: Dict) -> bool:
        """Test position tracking during incremental generation"""
        
        initial_seq = scenario["initial_seq"]
        generation_steps = scenario["generation_steps"]
        batch_size = scenario["batch_size"]
        
        tracking_consistent = True
        step_details = []
        
        # Simulate generation process
        for step in range(generation_steps):
            current_step = step + 1  # 1-indexed
            
            # Current position offset (initial + generated tokens)
            current_offset = initial_seq + step
            
            # For generation: we generate one token at a time
            new_token_seq_len = 1
            
            # Generate position IDs for new token
            jax_positions = self.create_position_ids_jax(batch_size, new_token_seq_len, current_offset)
            torch_positions = self.create_position_ids_torch(batch_size, new_token_seq_len, current_offset)
            
            # Compare
            pos_diff = np.max(np.abs(np.array(jax_positions) - torch_positions))
            step_consistent = pos_diff == 0
            tracking_consistent = tracking_consistent and step_consistent
            
            # Verify the position value is exactly what we expect
            expected_position = current_offset
            jax_pos_val = int(jax_positions[0, 0])
            torch_pos_val = int(torch_positions[0, 0])
            
            value_correct = (jax_pos_val == expected_position and torch_pos_val == expected_position)
            tracking_consistent = tracking_consistent and value_correct
            
            step_details.append(f"step{current_step}(offset={current_offset},pos={jax_pos_val},diff={pos_diff})")
        
        details = (f"initial_seq={initial_seq}, steps={generation_steps}, batch={batch_size}: "
                  + ", ".join(step_details))
        
        return tracking_consistent, details
    
    def test_rope_computation_alignment(self, batch_size: int, seq_len: int, offset: int = 0) -> bool:
        """Test RoPE computation alignment using position IDs"""
        
        # Create position IDs
        jax_positions = self.create_position_ids_jax(batch_size, seq_len, offset)
        torch_positions = self.create_position_ids_torch(batch_size, seq_len, offset)
        
        # Compute RoPE cos/sin
        jax_cos, jax_sin = self.compute_cos_sin_jax(jax_positions, self.head_dim, self.rope_theta)
        torch_cos, torch_sin = self.compute_cos_sin_torch(torch_positions, self.head_dim, self.rope_theta)
        
        # Compare results
        cos_diff = np.max(np.abs(np.array(jax_cos) - torch_cos))
        sin_diff = np.max(np.abs(np.array(jax_sin) - torch_sin))
        
        cos_shapes_match = jax_cos.shape == torch_cos.shape
        sin_shapes_match = jax_sin.shape == torch_sin.shape
        
        # Expected shape for cos/sin
        expected_rope_shape = (batch_size, seq_len, self.head_dim)
        cos_shape_correct = (jax_cos.shape == expected_rope_shape and torch_cos.shape == expected_rope_shape)
        sin_shape_correct = (jax_sin.shape == expected_rope_shape and torch_sin.shape == expected_rope_shape)
        
        all_passed = (cos_diff < self.rope_precision_threshold and sin_diff < self.rope_precision_threshold and
                     cos_shapes_match and sin_shapes_match and cos_shape_correct and sin_shape_correct)
        
        details = (f"batch={batch_size}, seq={seq_len}, offset={offset}: "
                  f"cos_diff={cos_diff:.2e}, sin_diff={sin_diff:.2e}, "
                  f"shapes_match={cos_shapes_match and sin_shapes_match}")
        
        return all_passed, details
    
    def test_position_sequence_consistency(self, scenario: Dict) -> bool:
        """Test position sequence consistency across generation"""
        
        initial_seq = scenario["initial_seq"]
        generation_steps = scenario["generation_steps"]
        batch_size = scenario["batch_size"]
        
        # Generate complete sequence positions for comparison
        total_seq_len = initial_seq + generation_steps
        complete_jax_positions = self.create_position_ids_jax(batch_size, total_seq_len, 0)
        complete_torch_positions = self.create_position_ids_torch(batch_size, total_seq_len, 0)
        
        # Verify complete sequence is consistent
        complete_diff = np.max(np.abs(np.array(complete_jax_positions) - complete_torch_positions))
        complete_consistent = complete_diff == 0
        
        # Now verify incremental generation produces the same sequence
        incremental_positions = []
        
        # Initial positions
        initial_positions = self.create_position_ids_jax(batch_size, initial_seq, 0)
        incremental_positions.append(initial_positions)
        
        # Generated positions
        for step in range(generation_steps):
            offset = initial_seq + step
            new_position = self.create_position_ids_jax(batch_size, 1, offset)
            incremental_positions.append(new_position)
        
        # Reconstruct complete sequence from incremental parts
        reconstructed_positions = jnp.concatenate(incremental_positions, axis=1)
        
        # Compare reconstructed vs complete
        reconstruction_diff = np.max(np.abs(np.array(reconstructed_positions) - np.array(complete_jax_positions)))
        reconstruction_consistent = reconstruction_diff == 0
        
        all_consistent = complete_consistent and reconstruction_consistent
        
        details = (f"initial_seq={initial_seq}, steps={generation_steps}, batch={batch_size}: "
                  f"complete_diff={complete_diff}, reconstruction_diff={reconstruction_diff}")
        
        return all_consistent, details
    
    def test_position_edge_cases(self) -> bool:
        """Test position ID edge cases"""
        
        edge_tests = []
        all_edge_correct = True
        
        # Edge case 1: Single token, zero offset
        edge1_passed, edge1_details = self.test_position_id_creation_alignment(1, 1, 0)
        all_edge_correct = all_edge_correct and edge1_passed
        edge_tests.append(f"single_token_zero_offset({edge1_passed})")
        
        # Edge case 2: Large offset
        edge2_passed, edge2_details = self.test_position_id_creation_alignment(1, 5, 1000)
        all_edge_correct = all_edge_correct and edge2_passed
        edge_tests.append(f"large_offset({edge2_passed})")
        
        # Edge case 3: Large batch
        edge3_passed, edge3_details = self.test_position_id_creation_alignment(10, 3, 5)
        all_edge_correct = all_edge_correct and edge3_passed
        edge_tests.append(f"large_batch({edge3_passed})")
        
        # Edge case 4: Maximum reasonable sequence length
        edge4_passed, edge4_details = self.test_position_id_creation_alignment(1, 100, 50)
        all_edge_correct = all_edge_correct and edge4_passed
        edge_tests.append(f"max_seq_len({edge4_passed})")
        
        details = "Edge cases: " + ", ".join(edge_tests)
        
        return all_edge_correct, details
    
    def run_position_id_validation(self) -> bool:
        """Run comprehensive position ID validation tests"""
        
        logger.info("🚀 Starting Phase 5B.2: Position ID Management")
        logger.info("Target: Perfect positional encoding alignment during generation")
        
        # Test 1: Position ID creation alignment
        logger.info("📊 Testing position ID creation alignment...")
        basic_test_cases = [(1, 1, 0), (1, 5, 0), (2, 3, 5), (1, 10, 0)]
        for batch_size, seq_len, offset in basic_test_cases:
            passed, details = self.test_position_id_creation_alignment(batch_size, seq_len, offset)
            self.log_validation_result(f"Position ID creation", passed, details)
        
        # Test 2: Incremental position tracking
        logger.info("📊 Testing incremental position tracking...")
        for scenario in self.test_scenarios:
            passed, details = self.test_incremental_position_tracking(scenario)
            self.log_validation_result(f"Incremental tracking", passed, details)
        
        # Test 3: RoPE computation alignment
        logger.info("📊 Testing RoPE computation alignment...")
        rope_test_cases = [(1, 1, 0), (1, 5, 10), (2, 3, 0)]
        for batch_size, seq_len, offset in rope_test_cases:
            passed, details = self.test_rope_computation_alignment(batch_size, seq_len, offset)
            self.log_validation_result(f"RoPE computation", passed, details)
        
        # Test 4: Position sequence consistency
        logger.info("📊 Testing position sequence consistency...")
        for scenario in self.test_scenarios[:2]:  # Use first 2 scenarios for sequence tests
            passed, details = self.test_position_sequence_consistency(scenario)
            self.log_validation_result(f"Sequence consistency", passed, details)
        
        # Test 5: Position edge cases
        logger.info("📊 Testing position edge cases...")
        passed, details = self.test_position_edge_cases()
        self.log_validation_result(f"Edge cases", passed, details)
        
        # Final validation summary
        if self.validation_passed:
            logger.info("🎉 Phase 5B.2 PASSED: Perfect positional encoding alignment achieved!")
            logger.info("✅ Position ID creation alignment verified")
            logger.info("✅ Incremental position tracking validated")
            logger.info("✅ RoPE computation alignment confirmed")
            logger.info("✅ Position sequence consistency tested")
            logger.info("✅ Edge cases handled correctly")
        else:
            logger.error("❌ Phase 5B.2 FAILED: Position ID management issues detected")
            logger.error("🔧 Review position tracking and RoPE computation")
        
        return self.validation_passed

def main():
    """Main Phase 5B.2 validation entry point"""
    
    print("="*80)
    print("PHASE 5B.2: POSITION ID MANAGEMENT")
    print("Building on Phase 3 RoPE expertise for perfect positional encoding")
    print("="*80)
    
    validator = Phase5B2PositionIDValidator()
    success = validator.run_position_id_validation()
    
    print("\n" + "="*80)
    if success:
        print("🎉 PHASE 5B.2 VALIDATION COMPLETE - ALL TESTS PASSED")
        print("✅ Perfect positional encoding alignment during generation")
    else:
        print("❌ PHASE 5B.2 VALIDATION FAILED")
        print("🔧 Position ID management requires improvements")
    print("="*80)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 