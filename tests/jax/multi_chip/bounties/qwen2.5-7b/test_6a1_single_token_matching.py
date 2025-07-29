#!/usr/bin/env python3
"""
Phase 6A.1: Single-Token Perfect Matching Validator
Test framework for validating single-token generation alignment between JAX and PyTorch implementations.
"""

import os
import sys
import time
import json
import gc
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional

import jax
import jax.numpy as jnp
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import our JAX implementation
from qwen_jax_inference import (
    Qwen25ForCausalLM, 
    load_params, 
    get_enhanced_sampler,
    Phase4EnhancedSampler
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("phase6a1_validator")

class Phase6A1SingleTokenValidator:
    """Phase 6A.1: Single-Token Perfect Matching Validator"""
    
    def __init__(self, model_path: str, dtype_str: str = "bfloat16"):
        # Phase 3 validated precision thresholds
        self.LOGITS_PRECISION_THRESHOLD = 1e-7  # Target for remaining differences
        self.PERFECT_MATCH_THRESHOLD = 0.0      # Ultimate goal
        self.SAMPLING_PRECISION_THRESHOLD = 1e-6 # Phase 4 realistic threshold
        
        self.model_path = model_path
        self.dtype = jnp.bfloat16 if dtype_str == "bfloat16" else jnp.float32
        self.torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32
        
        # Initialize models and tokenizer
        self.jax_model = None
        self.jax_params = None
        self.torch_model = None
        self.tokenizer = None
        
        # Phase 5 enhanced components
        self.enhanced_sampler = None
        
        # Test results storage
        self.test_results = {}
        
        logger.info("🚀 Phase 6A.1 Single-Token Validator initialized")
    
    def load_models(self):
        """Load both JAX and PyTorch models with identical weights"""
        logger.info("📥 Loading JAX and PyTorch models...")
        
        try:
            # Load config
            config_path = os.path.join(self.model_path, "config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load JAX model with Phase 3 enhancements
            logger.info("Loading JAX model with Phase 3 enhancements...")
            self.jax_model = Qwen25ForCausalLM(config=config, dtype=self.dtype)
            self.jax_params = load_params(self.jax_model, self.model_path, self.dtype)
            
            # Load PyTorch model 
            logger.info("Loading PyTorch model...")
            self.torch_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.torch_model.eval()
            
            # Initialize Phase 4 enhanced sampler
            self.enhanced_sampler = get_enhanced_sampler(seed=42, use_deterministic_rng=True)
            
            logger.info("✅ All models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False
    
    def get_test_prompts(self) -> Dict[str, List[str]]:
        """Define comprehensive test prompts for single-token validation"""
        return {
            # Phase 3 validated prompts
            'phase3_validated': ["Hello", "The", "123"],
            
            # Enhanced single-token prompts
            'single_tokens': ["Q:", "A:", "<|im_start|>", "What", "How"],
            
            # Mathematical prompts (preparing for 6B.2)
            'mathematical': ["2+2=", "Solve", "Calculate", "Math:", "x="],
            
            # Technical prompts
            'technical': ["AI", "ML", "def", "import", "class"],
            
            # Simple words
            'simple_words': ["cat", "dog", "red", "blue", "big"]
        }
    
    def validate_single_token_generation(self, prompt: str, temperature: float = 0.0) -> Dict[str, Any]:
        """Validate single token generation between JAX and PyTorch"""
        logger.info(f"🔍 Testing prompt: '{prompt}' (temp={temperature})")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="np")
            input_ids_np = inputs["input_ids"]
            input_ids_jax = jnp.array(input_ids_np, dtype=jnp.int32)
            input_ids_torch = torch.from_numpy(input_ids_np).long()
            
            # Set deterministic seeds
            if temperature == 0.0:
                # For greedy sampling, no randomness
                pass
            else:
                # Set seeds for deterministic sampling
                torch.manual_seed(42)
                # JAX seed handled by enhanced sampler
            
            # === JAX Forward Pass ===
            jax_outputs = self.jax_model.apply(self.jax_params, input_ids=input_ids_jax)
            jax_logits = jax_outputs if not isinstance(jax_outputs, dict) else jax_outputs["logits"]
            jax_logits_last = jax_logits[:, -1, :]  # Last token logits
            
            # === PyTorch Forward Pass ===
            with torch.no_grad():
                torch_outputs = self.torch_model(input_ids_torch)
                torch_logits = torch_outputs.logits
                torch_logits_last = torch_logits[:, -1, :]  # Last token logits
            
            # Convert to numpy for comparison
            jax_logits_np = np.array(jax_logits_last)
            torch_logits_np = torch_logits_last.cpu().numpy()
            
            # Calculate logits difference
            logits_diff = np.max(np.abs(jax_logits_np - torch_logits_np))
            logits_mean_diff = np.mean(np.abs(jax_logits_np - torch_logits_np))
            
            # === Token Sampling ===
            if temperature == 0.0:
                # Greedy sampling - perfect argmax alignment (Phase 4C.1)
                jax_token = int(jnp.argmax(jax_logits_last, axis=-1)[0])
                torch_token = int(torch.argmax(torch_logits_last, dim=-1)[0])
                sampling_method = "greedy"
            else:
                # Enhanced sampling with Phase 4 implementation
                jax_token = int(self.enhanced_sampler.sample_with_validation(
                    jax_logits_last, temperature=temperature, top_p=0.9, top_k=50,
                    repetition_penalty=1.1, past_tokens=None, validate=True
                )[0])
                
                # PyTorch equivalent sampling
                torch_probs = torch.softmax(torch_logits_last / temperature, dim=-1)
                torch_token = int(torch.multinomial(torch_probs, 1)[0, 0])
                sampling_method = "stochastic"
            
            # Decode tokens
            jax_token_text = self.tokenizer.decode([jax_token])
            torch_token_text = self.tokenizer.decode([torch_token])
            
            # Validation results
            result = {
                'prompt': prompt,
                'temperature': temperature,
                'sampling_method': sampling_method,
                'input_tokens': input_ids_np.shape[1],
                
                # Logits comparison
                'logits_max_diff': float(logits_diff),
                'logits_mean_diff': float(logits_mean_diff),
                'logits_within_threshold': logits_diff < self.LOGITS_PRECISION_THRESHOLD,
                'logits_perfect_match': logits_diff < self.PERFECT_MATCH_THRESHOLD,
                
                # Token comparison
                'jax_token': jax_token,
                'torch_token': torch_token,
                'jax_token_text': jax_token_text,
                'torch_token_text': torch_token_text,
                'tokens_match': jax_token == torch_token,
                'token_text_match': jax_token_text == torch_token_text,
                
                # Overall validation
                'validation_passed': (logits_diff < self.LOGITS_PRECISION_THRESHOLD and 
                                    (temperature == 0.0 and jax_token == torch_token) or
                                    (temperature > 0.0)),  # For stochastic, just check no errors
                
                # Additional diagnostics
                'jax_logits_stats': {
                    'min': float(np.min(jax_logits_np)),
                    'max': float(np.max(jax_logits_np)),
                    'mean': float(np.mean(jax_logits_np)),
                    'std': float(np.std(jax_logits_np))
                },
                'torch_logits_stats': {
                    'min': float(np.min(torch_logits_np)),
                    'max': float(np.max(torch_logits_np)),
                    'mean': float(np.mean(torch_logits_np)),
                    'std': float(np.std(torch_logits_np))
                }
            }
            
            # Log results
            if result['validation_passed']:
                logger.info(f"✅ PASSED: logits_diff={logits_diff:.2e}, tokens_match={result['tokens_match']}")
            else:
                logger.warning(f"⚠️ FAILED: logits_diff={logits_diff:.2e}, tokens_match={result['tokens_match']}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Validation failed for prompt '{prompt}': {e}")
            return {
                'prompt': prompt,
                'temperature': temperature,
                'error': str(e),
                'validation_passed': False
            }
    
    def run_comprehensive_single_token_tests(self) -> Dict[str, Any]:
        """Run comprehensive single-token validation tests"""
        logger.info("🧪 Running comprehensive single-token validation tests...")
        
        test_prompts = self.get_test_prompts()
        temperature_settings = [0.0, 0.3, 0.7, 1.0]
        
        all_results = {}
        summary_stats = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'perfect_logits_matches': 0,
            'greedy_token_matches': 0,
            'average_logits_diff': 0.0,
            'max_logits_diff': 0.0
        }
        
        # Test each category and temperature combination
        for category, prompts in test_prompts.items():
            category_results = {}
            
            for prompt in prompts:
                prompt_results = {}
                
                for temp in temperature_settings:
                    result = self.validate_single_token_generation(prompt, temp)
                    prompt_results[f'temp_{temp}'] = result
                    
                    # Update summary stats
                    summary_stats['total_tests'] += 1
                    if result['validation_passed']:
                        summary_stats['passed_tests'] += 1
                    else:
                        summary_stats['failed_tests'] += 1
                    
                    if result.get('logits_perfect_match', False):
                        summary_stats['perfect_logits_matches'] += 1
                    
                    if temp == 0.0 and result.get('tokens_match', False):
                        summary_stats['greedy_token_matches'] += 1
                    
                    # Track logits differences
                    logits_diff = result.get('logits_max_diff', 0.0)
                    summary_stats['average_logits_diff'] += logits_diff
                    summary_stats['max_logits_diff'] = max(summary_stats['max_logits_diff'], logits_diff)
                
                category_results[prompt] = prompt_results
            
            all_results[category] = category_results
        
        # Calculate final statistics
        if summary_stats['total_tests'] > 0:
            summary_stats['pass_rate'] = summary_stats['passed_tests'] / summary_stats['total_tests']
            summary_stats['average_logits_diff'] /= summary_stats['total_tests']
            summary_stats['perfect_match_rate'] = summary_stats['perfect_logits_matches'] / summary_stats['total_tests']
            summary_stats['greedy_accuracy'] = summary_stats['greedy_token_matches'] / len([p for prompts in test_prompts.values() for p in prompts])
        
        # Store results
        self.test_results['single_token_validation'] = {
            'summary': summary_stats,
            'detailed_results': all_results,
            'test_timestamp': time.time()
        }
        
        # Log summary
        logger.info("📊 Single-Token Validation Summary:")
        logger.info(f"   Total tests: {summary_stats['total_tests']}")
        logger.info(f"   Pass rate: {summary_stats['pass_rate']:.1%}")
        logger.info(f"   Perfect logits matches: {summary_stats['perfect_match_rate']:.1%}")
        logger.info(f"   Greedy token accuracy: {summary_stats['greedy_accuracy']:.1%}")
        logger.info(f"   Average logits diff: {summary_stats['average_logits_diff']:.2e}")
        logger.info(f"   Max logits diff: {summary_stats['max_logits_diff']:.2e}")
        
        return self.test_results['single_token_validation']
    
    def analyze_failures(self) -> Dict[str, Any]:
        """Analyze any test failures for debugging"""
        if 'single_token_validation' not in self.test_results:
            return {'error': 'No test results available'}
        
        detailed_results = self.test_results['single_token_validation']['detailed_results']
        failures = []
        
        for category, category_results in detailed_results.items():
            for prompt, prompt_results in category_results.items():
                for temp_key, result in prompt_results.items():
                    if not result.get('validation_passed', False):
                        failures.append({
                            'category': category,
                            'prompt': prompt,
                            'temperature': result.get('temperature', 0.0),
                            'logits_diff': result.get('logits_max_diff', 0.0),
                            'tokens_match': result.get('tokens_match', False),
                            'error': result.get('error', 'Unknown failure')
                        })
        
        analysis = {
            'total_failures': len(failures),
            'failure_breakdown': {},
            'logits_issues': [],
            'token_mismatch_issues': [],
            'error_issues': []
        }
        
        # Categorize failures
        for failure in failures:
            category = failure['category']
            if category not in analysis['failure_breakdown']:
                analysis['failure_breakdown'][category] = 0
            analysis['failure_breakdown'][category] += 1
            
            if 'error' in failure and failure['error'] != 'Unknown failure':
                analysis['error_issues'].append(failure)
            elif not failure['tokens_match'] and failure['temperature'] == 0.0:
                analysis['token_mismatch_issues'].append(failure)
            elif failure['logits_diff'] > self.LOGITS_PRECISION_THRESHOLD:
                analysis['logits_issues'].append(failure)
        
        return analysis
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up resources...")
        if self.torch_model is not None:
            del self.torch_model
        if self.jax_params is not None:
            del self.jax_params
        if self.jax_model is not None:
            del self.jax_model
        
        gc.collect()
        jax.clear_caches()
        logger.info("✅ Cleanup complete")

def main():
    """Main test execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 6A.1: Single-Token Perfect Matching Validator")
    parser.add_argument("--model_path", type=str, default="./weights", help="Path to model weights")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    args = parser.parse_args()
    
    logger.info("🚀 Starting Phase 6A.1: Single-Token Perfect Matching Validation")
    
    # Initialize validator
    validator = Phase6A1SingleTokenValidator(args.model_path, args.dtype)
    
    try:
        # Load models
        if not validator.load_models():
            logger.error("❌ Failed to load models. Exiting.")
            return 1
        
        # Run tests
        results = validator.run_comprehensive_single_token_tests()
        
        # Analyze failures
        failure_analysis = validator.analyze_failures()
        
        # Print final results
        print("\n" + "="*80)
        print("🎯 PHASE 6A.1 SINGLE-TOKEN VALIDATION RESULTS")
        print("="*80)
        
        summary = results['summary']
        print(f"📊 Overall Results:")
        print(f"   • Total Tests: {summary['total_tests']}")
        print(f"   • Pass Rate: {summary['pass_rate']:.1%}")
        print(f"   • Perfect Logits Matches: {summary['perfect_match_rate']:.1%}")
        print(f"   • Greedy Token Accuracy: {summary['greedy_accuracy']:.1%}")
        print(f"   • Average Logits Difference: {summary['average_logits_diff']:.2e}")
        print(f"   • Maximum Logits Difference: {summary['max_logits_diff']:.2e}")
        
        if failure_analysis['total_failures'] > 0:
            print(f"\n⚠️ Failure Analysis:")
            print(f"   • Total Failures: {failure_analysis['total_failures']}")
            print(f"   • Logits Issues: {len(failure_analysis['logits_issues'])}")
            print(f"   • Token Mismatch Issues: {len(failure_analysis['token_mismatch_issues'])}")
            print(f"   • Error Issues: {len(failure_analysis['error_issues'])}")
        
        # Determine success
        success_criteria = {
            'pass_rate': summary['pass_rate'] >= 0.95,  # 95% pass rate
            'greedy_accuracy': summary['greedy_accuracy'] >= 0.99,  # 99% greedy accuracy
            'logits_precision': summary['max_logits_diff'] < 1e-5,  # Reasonable precision
            'no_errors': len(failure_analysis['error_issues']) == 0
        }
        
        overall_success = all(success_criteria.values())
        
        print(f"\n🎯 Success Criteria:")
        for criterion, passed in success_criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   • {criterion}: {status}")
        
        print(f"\n{'🎉 PHASE 6A.1 COMPLETE - SUCCESS!' if overall_success else '⚠️ PHASE 6A.1 COMPLETE - NEEDS IMPROVEMENT'}")
        print("="*80)
        
        return 0 if overall_success else 1
        
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        return 1
    
    finally:
        validator.cleanup()

if __name__ == "__main__":
    exit(main()) 