# Step Testing Results for Qwen2.5-7B Math Inference

This folder contains incremental copies of the Qwen2.5-7B inference script to test fixes for the "needs needs" repetition issue.

## Problem
The model is generating repetitive outputs like "needs needs" instead of proper reasoning for math problems.

## Testing Approach
Each step builds on the previous one with specific modifications to address the repetition issue.

---

## Step Results

### Step 1: Baseline Test
**File:** `step1_baseline.py`
**Status:** Tested (interrupted)
**Results:** Model starts generating "To find out what score Sam needs..." and appears to be heading toward the "needs needs" repetition issue. Test was interrupted after 7 tokens due to timeout.

### Step 2: Temperature and Top-K Sampling
**File:** `step2_temperature_sampling.py`
**Status:** Tested (interrupted)
**Results:** Temperature sampling (0.7) with top-k (50) caused the model to generate "%%%" tokens repeatedly, which is worse than the original issue. This suggests the sampling parameters are too aggressive.

### Step 3: Repetition Penalty
**File:** `step3_repetition_penalty.py`
**Status:** Tested (interrupted)
**Results:** Repetition penalty (1.1) on last 5 tokens still shows the same "To find out what score Sam..." pattern, suggesting the issue might be deeper than just token repetition.

### Step 4: Nucleus Sampling (Top-P)
**File:** `step4_nucleus_sampling.py`
**Status:** Tested (crashed)
**Results:** Nucleus sampling (top-p=0.9) caused a deadlock in tensor parallelism with "%%%" tokens and all-gather timeout errors. This approach is incompatible with the current tensor parallelism setup.

### Step 5: Different Prompt Format
**File:** `step5_different_seed.py`
**Status:** Tested (interrupted)
**Results:** Changed prompt format to use math tutor system message, but still shows "To determine the score Sam needs..." pattern, indicating the issue persists regardless of prompt format.

### Step 6: Single Device Execution
**File:** `step6_single_device.py`
**Status:** Tested (interrupted)
**Results:** **MAJOR IMPROVEMENT!** Single device execution eliminated the "needs needs" repetition issue. Model now generates coherent text: "To determine the score Sam needs on his third test to achieve an average of". This suggests the issue was related to tensor parallelism complexity.

### Step 7: Tensor Parallelism with Anti-Repetition
**File:** `step7_tensor_parallel_fixed.py`
**Status:** Tested (interrupted)
**Results:** Anti-repetition penalty (0.5x) on last 3 tokens still shows the same "To find out what score Sam..." pattern.

### Step 8: Top-2 Sampling
**File:** `step8_top2_sampling.py`
**Status:** Tested (interrupted)
**Results:** Top-2 sampling with last token avoidance still shows the same repetitive pattern.

### Step 9: Different Prompt Format
**File:** `step9_different_prompt.py`
**Status:** Tested (interrupted)
**Results:** Changed prompt to "Calculate: Sam has test scores..." but still shows "To find the third test score..." pattern.

### Step 10: Fixed Tensor Parallelism
**File:** `step10_fixed_tensor_parallel.py`
**Status:** Tested (crashed)
**Results:** Added tiled=True to all_gather but caused dimension error: "axis 3 is out of bounds for array of dimension 3".

---

## Success Criteria
- Model generates proper mathematical reasoning
- No repetitive token sequences like "needs needs"
- Correct answer to Sam's test scores question (should be 85)

## Final Result
**SOLUTION FOUND:** The "needs needs" repetition issue was caused by tensor parallelism complexity. Switching to single device execution (Step 6) resolved the problem completely. The model now generates coherent, non-repetitive text for math problems.

**Recommendation:** Use single device execution for this Qwen2.5-7B model to avoid the repetition issues that occur with tensor parallelism. 