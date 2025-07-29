#!/usr/bin/env python3
"""
Phase 6B: Mathematical Reasoning Validation
Test framework for validating mathematical reasoning quality between JAX and PyTorch implementations.
Focus on GSM8K-style math word problems and numerical answer correctness.
"""

import os
import sys
import time
import json
import gc
import re
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
    enhanced_generate
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("phase6b_math_validator")

class Phase6BMathematicalReasoningValidator:
    """Phase 6B: Mathematical Reasoning Quality Validator"""

    def __init__(self, model_path: str, dtype_str: str = "bfloat16"):
        # Phase 6B Success Criteria
        self.MATHEMATICAL_ACCURACY_THRESHOLD = 0.85  # >85% mathematical accuracy
        self.NUMERICAL_CORRECTNESS_THRESHOLD = 0.95  # >95% numerical answer correctness
        self.REASONING_QUALITY_THRESHOLD = 0.80     # >80% reasoning quality match

        self.model_path = model_path
        self.dtype = jnp.bfloat16 if dtype_str == "bfloat16" else jnp.float32
        self.torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32

        # Initialize models and tokenizer
        self.jax_model = None
        self.jax_params = None
        self.torch_model = None
        self.tokenizer = None

        # Generation parameters for mathematical reasoning
        self.generation_config = {
            'max_new_tokens': 512,
            'temperature': 0.1,  # Low temperature for mathematical consistency
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.1,
            'do_sample': True
        }

        # Test results storage
        self.test_results = {}

        logger.info("🚀 Phase 6B Mathematical Reasoning Validator initialized")

    def load_models(self):
        """Load both JAX and PyTorch models"""
        logger.info("📥 Loading JAX and PyTorch models for mathematical reasoning...")

        try:
            # Load config
            config_path = os.path.join(self.model_path, "config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            # Load JAX model
            logger.info("Loading JAX model...")
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

            logger.info("✅ All models loaded successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False

    def get_mathematical_test_problems(self) -> Dict[str, List[Dict[str, Any]]]:
        """Define comprehensive mathematical test problems"""
        return {
            # Basic arithmetic problems
            'basic_arithmetic': [
                {
                    'problem': "What is 23 + 47?",
                    'expected_answer': 70,
                    'type': 'addition'
                },
                {
                    'problem': "Calculate 156 - 89.",
                    'expected_answer': 67,
                    'type': 'subtraction'
                },
                {
                    'problem': "What is 12 × 15?",
                    'expected_answer': 180,
                    'type': 'multiplication'
                },
                {
                    'problem': "Calculate 144 ÷ 12.",
                    'expected_answer': 12,
                    'type': 'division'
                }
            ],

            # GSM8K-style word problems
            'word_problems': [
                {
                    'problem': "Sarah has 24 apples. She gives 8 apples to her friend and then buys 15 more apples. How many apples does Sarah have now?",
                    'expected_answer': 31,
                    'type': 'multi_step'
                },
                {
                    'problem': "A store sells pencils for $0.50 each. If Tom buys 6 pencils, how much money does he spend?",
                    'expected_answer': 3.0,
                    'type': 'multiplication_decimal'
                },
                {
                    'problem': "There are 180 students in a school. If 2/3 of them are boys, how many boys are there?",
                    'expected_answer': 120,
                    'type': 'fraction'
                },
                {
                    'problem': "A rectangle has a length of 15 cm and a width of 8 cm. What is its area?",
                    'expected_answer': 120,
                    'type': 'geometry'
                }
            ],

            # Multi-step reasoning problems
            'complex_reasoning': [
                {
                    'problem': "John earns $15 per hour. He works 8 hours a day for 5 days a week. How much money does he earn in 2 weeks?",
                    'expected_answer': 1200,
                    'type': 'multi_step_calculation'
                },
                {
                    'problem': "A pizza is cut into 8 equal slices. If 3 people each eat 2 slices, how many slices are left?",
                    'expected_answer': 2,
                    'type': 'subtraction_word_problem'
                },
                {
                    'problem': "The temperature was 22°C in the morning. It increased by 8°C during the day, then decreased by 5°C in the evening. What is the final temperature?",
                    'expected_answer': 25,
                    'type': 'sequential_operations'
                }
            ],

            # Percentage and ratio problems
            'percentage_problems': [
                {
                    'problem': "What is 25% of 80?",
                    'expected_answer': 20,
                    'type': 'percentage'
                },
                {
                    'problem': "If a shirt costs $40 and is on sale for 30% off, what is the sale price?",
                    'expected_answer': 28,
                    'type': 'percentage_discount'
                }
            ]
        }

    def format_math_prompt(self, problem: str) -> str:
        """Format mathematical problem using chat template"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant that solves mathematical problems step by step."},
            {"role": "user", "content": f"Solve this math problem step by step:\n\n{problem}\n\nShow your work and provide the final numerical answer."}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

    def extract_numerical_answer(self, response: str) -> Optional[float]:
        """Extract the final numerical answer from a response"""
        # Look for patterns like "The answer is X", "Final answer: X", "= X", etc.
        patterns = [
            r"(?:the\s+)?(?:final\s+)?answer\s+is\s+(\d+(?:\.\d+)?)",
            r"(?:final\s+)?answer:\s*(\d+(?:\.\d+)?)",
            r"=\s*(\d+(?:\.\d+)?)\s*$",
            r"(\d+(?:\.\d+)?)\s*(?:dollars?|apples?|students?|cm|°C|slices?|hours?)\s*$",
            r"(?:total|result|sum):\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*$"  # Last number in the response
        ]
        
        response_lower = response.lower().strip()
        
        for pattern in patterns:
            matches = re.findall(pattern, response_lower, re.MULTILINE | re.IGNORECASE)
            if matches:
                try:
                    return float(matches[-1])  # Take the last match
                except ValueError:
                    continue
        
        # If no pattern matches, try to find any number at the end
        numbers = re.findall(r'\d+(?:\.\d+)?', response)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        
        return None

    def evaluate_reasoning_quality(self, jax_response: str, torch_response: str) -> Dict[str, Any]:
        """Evaluate the quality of mathematical reasoning"""
        # Simple heuristics for reasoning quality
        
        # Check for step-by-step reasoning
        jax_has_steps = any(word in jax_response.lower() for word in ['step', 'first', 'then', 'next', 'finally'])
        torch_has_steps = any(word in torch_response.lower() for word in ['step', 'first', 'then', 'next', 'finally'])
        
        # Check for mathematical operations
        math_words = ['add', 'subtract', 'multiply', 'divide', 'calculate', 'total', 'sum']
        jax_math_words = sum(1 for word in math_words if word in jax_response.lower())
        torch_math_words = sum(1 for word in math_words if word in torch_response.lower())
        
        # Check response length (longer usually means more explanation)
        length_ratio = len(jax_response) / max(len(torch_response), 1)
        
        # Simple quality score
        quality_match = (
            (jax_has_steps == torch_has_steps) * 0.3 +
            (abs(jax_math_words - torch_math_words) <= 1) * 0.4 +
            (0.5 <= length_ratio <= 2.0) * 0.3
        )
        
        return {
            'quality_score': quality_match,
            'jax_has_steps': jax_has_steps,
            'torch_has_steps': torch_has_steps,
            'jax_math_words': jax_math_words,
            'torch_math_words': torch_math_words,
            'length_ratio': length_ratio
        }

    def test_mathematical_problem(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single mathematical problem with both implementations"""
        problem = problem_data['problem']
        expected_answer = problem_data['expected_answer']
        problem_type = problem_data['type']
        
        logger.info(f"🧮 Testing: {problem[:50]}...")
        
        try:
            # Format prompt
            formatted_prompt = self.format_math_prompt(problem)
            
            # Generate with JAX
            logger.info("   Generating JAX response...")
            jax_response = enhanced_generate(
                model=self.jax_model,
                params=self.jax_params,
                tokenizer=self.tokenizer,
                prompt=formatted_prompt,
                **self.generation_config
            )
            
            # Generate with PyTorch
            logger.info("   Generating PyTorch response...")
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            
            with torch.no_grad():
                torch_outputs = self.torch_model.generate(
                    inputs.input_ids,
                    max_new_tokens=self.generation_config['max_new_tokens'],
                    temperature=self.generation_config['temperature'],
                    top_p=self.generation_config['top_p'],
                    top_k=self.generation_config['top_k'],
                    repetition_penalty=self.generation_config['repetition_penalty'],
                    do_sample=self.generation_config['do_sample'],
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode PyTorch response
            torch_response = self.tokenizer.decode(
                torch_outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            # Extract numerical answers
            jax_answer = self.extract_numerical_answer(jax_response)
            torch_answer = self.extract_numerical_answer(torch_response)
            
            # Evaluate reasoning quality
            reasoning_eval = self.evaluate_reasoning_quality(jax_response, torch_response)
            
            # Check numerical correctness
            jax_correct = jax_answer is not None and abs(jax_answer - expected_answer) < 0.01
            torch_correct = torch_answer is not None and abs(torch_answer - expected_answer) < 0.01
            
            # Check if both answers match (even if wrong)
            answers_match = (
                jax_answer is not None and torch_answer is not None and 
                abs(jax_answer - torch_answer) < 0.01
            )
            
            result = {
                'problem': problem,
                'problem_type': problem_type,
                'expected_answer': expected_answer,
                
                # Responses
                'jax_response': jax_response,
                'torch_response': torch_response,
                
                # Extracted answers
                'jax_answer': jax_answer,
                'torch_answer': torch_answer,
                
                # Correctness
                'jax_correct': jax_correct,
                'torch_correct': torch_correct,
                'answers_match': answers_match,
                
                # Reasoning quality
                'reasoning_quality': reasoning_eval,
                
                # Overall assessment
                'both_correct': jax_correct and torch_correct,
                'jax_as_good_as_torch': jax_correct >= torch_correct,
                'test_passed': jax_correct and reasoning_eval['quality_score'] >= 0.5
            }
            
            # Log result
            status = "✅ PASS" if result['test_passed'] else "❌ FAIL"
            logger.info(f"   {status}: JAX={jax_answer}, PyTorch={torch_answer}, Expected={expected_answer}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Test failed for problem: {e}")
            return {
                'problem': problem,
                'problem_type': problem_type,
                'expected_answer': expected_answer,
                'error': str(e),
                'test_passed': False
            }

    def run_comprehensive_mathematical_reasoning_tests(self) -> Dict[str, Any]:
        """Run comprehensive mathematical reasoning validation tests"""
        logger.info("🧮 Running comprehensive mathematical reasoning validation tests...")
        
        test_problems = self.get_mathematical_test_problems()
        
        all_results = {}
        summary_stats = {
            'total_problems': 0,
            'jax_correct': 0,
            'torch_correct': 0,
            'both_correct': 0,
            'answers_match': 0,
            'jax_as_good_as_torch': 0,
            'tests_passed': 0,
            'average_reasoning_quality': 0.0,
            'by_category': {}
        }
        
        # Test each category
        for category, problems in test_problems.items():
            logger.info(f"\n📂 Testing category: {category}")
            category_results = []
            category_stats = {
                'total': len(problems),
                'jax_correct': 0,
                'torch_correct': 0,
                'both_correct': 0,
                'tests_passed': 0
            }
            
            for problem_data in problems:
                result = self.test_mathematical_problem(problem_data)
                category_results.append(result)
                
                # Update category stats
                if result.get('jax_correct', False):
                    category_stats['jax_correct'] += 1
                if result.get('torch_correct', False):
                    category_stats['torch_correct'] += 1
                if result.get('both_correct', False):
                    category_stats['both_correct'] += 1
                if result.get('test_passed', False):
                    category_stats['tests_passed'] += 1
                
                # Update overall stats
                summary_stats['total_problems'] += 1
                if result.get('jax_correct', False):
                    summary_stats['jax_correct'] += 1
                if result.get('torch_correct', False):
                    summary_stats['torch_correct'] += 1
                if result.get('both_correct', False):
                    summary_stats['both_correct'] += 1
                if result.get('answers_match', False):
                    summary_stats['answers_match'] += 1
                if result.get('jax_as_good_as_torch', False):
                    summary_stats['jax_as_good_as_torch'] += 1
                if result.get('test_passed', False):
                    summary_stats['tests_passed'] += 1
                if 'reasoning_quality' in result:
                    summary_stats['average_reasoning_quality'] += result['reasoning_quality']['quality_score']
            
            # Calculate category percentages
            category_stats['jax_accuracy'] = category_stats['jax_correct'] / category_stats['total']
            category_stats['torch_accuracy'] = category_stats['torch_correct'] / category_stats['total']
            category_stats['pass_rate'] = category_stats['tests_passed'] / category_stats['total']
            
            all_results[category] = {
                'problems': category_results,
                'stats': category_stats
            }
            summary_stats['by_category'][category] = category_stats
            
            # Log category summary
            logger.info(f"   📊 {category}: JAX {category_stats['jax_accuracy']:.1%}, PyTorch {category_stats['torch_accuracy']:.1%}, Pass {category_stats['pass_rate']:.1%}")
        
        # Calculate final statistics
        if summary_stats['total_problems'] > 0:
            summary_stats['jax_accuracy'] = summary_stats['jax_correct'] / summary_stats['total_problems']
            summary_stats['torch_accuracy'] = summary_stats['torch_correct'] / summary_stats['total_problems']
            summary_stats['numerical_correctness'] = summary_stats['jax_correct'] / summary_stats['total_problems']
            summary_stats['answer_match_rate'] = summary_stats['answers_match'] / summary_stats['total_problems']
            summary_stats['jax_quality_rate'] = summary_stats['jax_as_good_as_torch'] / summary_stats['total_problems']
            summary_stats['overall_pass_rate'] = summary_stats['tests_passed'] / summary_stats['total_problems']
            summary_stats['average_reasoning_quality'] /= summary_stats['total_problems']
        
        # Store results
        self.test_results['mathematical_reasoning'] = {
            'summary': summary_stats,
            'detailed_results': all_results,
            'test_timestamp': time.time()
        }
        
        # Log comprehensive summary
        logger.info("\n📊 Mathematical Reasoning Validation Summary:")
        logger.info(f"   Total problems tested: {summary_stats['total_problems']}")
        logger.info(f"   JAX accuracy: {summary_stats['jax_accuracy']:.1%}")
        logger.info(f"   PyTorch accuracy: {summary_stats['torch_accuracy']:.1%}")
        logger.info(f"   JAX numerical correctness: {summary_stats['numerical_correctness']:.1%}")
        logger.info(f"   Answer match rate: {summary_stats['answer_match_rate']:.1%}")
        logger.info(f"   JAX quality rate: {summary_stats['jax_quality_rate']:.1%}")
        logger.info(f"   Overall pass rate: {summary_stats['overall_pass_rate']:.1%}")
        logger.info(f"   Average reasoning quality: {summary_stats['average_reasoning_quality']:.1%}")
        
        return self.test_results['mathematical_reasoning']

    def analyze_mathematical_failures(self) -> Dict[str, Any]:
        """Analyze mathematical reasoning failures for debugging"""
        if 'mathematical_reasoning' not in self.test_results:
            return {'error': 'No test results available'}
        
        detailed_results = self.test_results['mathematical_reasoning']['detailed_results']
        failures = []
        success_patterns = []
        
        for category, category_data in detailed_results.items():
            for result in category_data['problems']:
                if not result.get('test_passed', False):
                    failures.append({
                        'category': category,
                        'problem': result['problem'][:100] + "...",
                        'expected': result.get('expected_answer'),
                        'jax_answer': result.get('jax_answer'),
                        'torch_answer': result.get('torch_answer'),
                        'jax_correct': result.get('jax_correct', False),
                        'torch_correct': result.get('torch_correct', False),
                        'error': result.get('error', 'Unknown')
                    })
                else:
                    success_patterns.append({
                        'category': category,
                        'problem_type': result.get('problem_type'),
                        'reasoning_quality': result.get('reasoning_quality', {}).get('quality_score', 0)
                    })
        
        analysis = {
            'total_failures': len(failures),
            'total_successes': len(success_patterns),
            'failure_by_category': {},
            'common_failure_patterns': [],
            'success_patterns': success_patterns[:5]  # Top 5 successes
        }
        
        # Categorize failures
        for failure in failures:
            category = failure['category']
            if category not in analysis['failure_by_category']:
                analysis['failure_by_category'][category] = []
            analysis['failure_by_category'][category].append(failure)
        
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
    
    parser = argparse.ArgumentParser(description="Phase 6B: Mathematical Reasoning Validation")
    parser.add_argument("--model_path", type=str, default="./weights", help="Path to model weights")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    args = parser.parse_args()
    
    logger.info("🚀 Starting Phase 6B: Mathematical Reasoning Validation")
    
    # Initialize validator
    validator = Phase6BMathematicalReasoningValidator(args.model_path, args.dtype)
    
    try:
        # Load models
        if not validator.load_models():
            logger.error("❌ Failed to load models. Exiting.")
            return 1
        
        # Run mathematical reasoning tests
        results = validator.run_comprehensive_mathematical_reasoning_tests()
        
        # Analyze failures
        failure_analysis = validator.analyze_mathematical_failures()
        
        # Print final results
        print("\n" + "="*80)
        print("🧮 PHASE 6B MATHEMATICAL REASONING VALIDATION RESULTS")
        print("="*80)
        
        summary = results['summary']
        print(f"📊 Overall Mathematical Performance:")
        print(f"   • Total Problems Tested: {summary['total_problems']}")
        print(f"   • JAX Mathematical Accuracy: {summary['jax_accuracy']:.1%}")
        print(f"   • PyTorch Mathematical Accuracy: {summary['torch_accuracy']:.1%}")
        print(f"   • JAX Numerical Correctness: {summary['numerical_correctness']:.1%}")
        print(f"   • Answer Match Rate: {summary['answer_match_rate']:.1%}")
        print(f"   • JAX Quality Rate: {summary['jax_quality_rate']:.1%}")
        print(f"   • Overall Pass Rate: {summary['overall_pass_rate']:.1%}")
        print(f"   • Average Reasoning Quality: {summary['average_reasoning_quality']:.1%}")
        
        print(f"\n📂 By Category:")
        for category, stats in summary['by_category'].items():
            print(f"   • {category}: JAX {stats['jax_accuracy']:.1%}, PyTorch {stats['torch_accuracy']:.1%}, Pass {stats['pass_rate']:.1%}")
        
        if failure_analysis['total_failures'] > 0:
            print(f"\n⚠️ Failure Analysis:")
            print(f"   • Total Failures: {failure_analysis['total_failures']}")
            for category, failures in failure_analysis['failure_by_category'].items():
                print(f"   • {category}: {len(failures)} failures")
        
        # Determine success criteria
        success_criteria = {
            'mathematical_accuracy': summary['jax_accuracy'] >= validator.MATHEMATICAL_ACCURACY_THRESHOLD,
            'numerical_correctness': summary['numerical_correctness'] >= validator.NUMERICAL_CORRECTNESS_THRESHOLD,
            'reasoning_quality': summary['average_reasoning_quality'] >= validator.REASONING_QUALITY_THRESHOLD,
            'jax_as_good_as_torch': summary['jax_quality_rate'] >= 0.90  # JAX should be as good as PyTorch 90% of the time
        }
        
        overall_success = all(success_criteria.values())
        
        print(f"\n🎯 Success Criteria (Phase 6B):")
        print(f"   • Mathematical Accuracy ≥{validator.MATHEMATICAL_ACCURACY_THRESHOLD:.0%}: {'✅ PASS' if success_criteria['mathematical_accuracy'] else '❌ FAIL'}")
        print(f"   • Numerical Correctness ≥{validator.NUMERICAL_CORRECTNESS_THRESHOLD:.0%}: {'✅ PASS' if success_criteria['numerical_correctness'] else '❌ FAIL'}")
        print(f"   • Reasoning Quality ≥{validator.REASONING_QUALITY_THRESHOLD:.0%}: {'✅ PASS' if success_criteria['reasoning_quality'] else '❌ FAIL'}")
        print(f"   • JAX as Good as PyTorch ≥90%: {'✅ PASS' if success_criteria['jax_as_good_as_torch'] else '❌ FAIL'}")
        
        print(f"\n{'🎉 PHASE 6B COMPLETE - MATHEMATICAL REASONING SUCCESS!' if overall_success else '⚠️ PHASE 6B COMPLETE - NEEDS IMPROVEMENT'}")
        print("="*80)
        
        return 0 if overall_success else 1
        
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        return 1
        
    finally:
        validator.cleanup()

if __name__ == "__main__":
    exit(main()) 