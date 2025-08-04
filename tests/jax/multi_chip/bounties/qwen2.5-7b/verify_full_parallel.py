#!/usr/bin/env python3
"""
Verification script to confirm full tensor parallelism by checking source code.
"""

import ast
import os

def verify_full_parallel():
    """Verify that all attention projections use ParallelDense by checking source code."""
    
    print("=== VERIFYING FULL TENSOR PARALLELISM ===")
    
    # Read the source code
    with open("q25j7_tensor_parallel_clean.py", "r") as f:
        source_code = f.read()
    
    # Parse the AST
    tree = ast.parse(source_code)
    
    # Find the FullyParallelQwenAttention class
    attention_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "FullyParallelQwenAttention":
            attention_class = node
            break
    
    if not attention_class:
        print("❌ Could not find FullyParallelQwenAttention class")
        return False
    
    print(f"Found FullyParallelQwenAttention class")
    
    # Check the setup method for ParallelDense usage
    setup_method = None
    for node in ast.walk(attention_class):
        if isinstance(node, ast.FunctionDef) and node.name == "setup":
            setup_method = node
            break
    
    if not setup_method:
        print("❌ Could not find setup method in FullyParallelQwenAttention")
        return False
    
    # Look for ParallelDense assignments
    parallel_dense_assignments = []
    for node in ast.walk(setup_method):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'ParallelDense':
                        parallel_dense_assignments.append(target.attr)
    
    print(f"\n--- Attention Projections ---")
    expected_projections = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    all_parallel = True
    
    for proj in expected_projections:
        if proj in parallel_dense_assignments:
            print(f"  ✅ {proj}: ParallelDense")
        else:
            print(f"  ❌ {proj}: NOT ParallelDense")
            all_parallel = False
    
    # Check QwenMLP class
    mlp_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "QwenMLP":
            mlp_class = node
            break
    
    if mlp_class:
        print(f"\n--- MLP Projections ---")
        mlp_setup = None
        for node in ast.walk(mlp_class):
            if isinstance(node, ast.FunctionDef) and node.name == "setup":
                mlp_setup = node
                break
        
        if mlp_setup:
            mlp_parallel_assignments = []
            for node in ast.walk(mlp_setup):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Attribute) and target.attr in ['gate_proj', 'up_proj', 'down_proj']:
                            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'ParallelDense':
                                mlp_parallel_assignments.append(target.attr)
            
            expected_mlp_projections = ['gate_proj', 'up_proj', 'down_proj']
            all_mlp_parallel = True
            
            for proj in expected_mlp_projections:
                if proj in mlp_parallel_assignments:
                    print(f"  ✅ {proj}: ParallelDense")
                else:
                    print(f"  ❌ {proj}: NOT ParallelDense")
                    all_mlp_parallel = False
        else:
            print("  ❌ Could not find MLP setup method")
            all_mlp_parallel = False
    else:
        print("  ❌ Could not find QwenMLP class")
        all_mlp_parallel = False
    
    # Check Qwen25ForCausalLM class
    main_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Qwen25ForCausalLM":
            main_class = node
            break
    
    if main_class:
        print(f"\n--- Main Model Components ---")
        main_setup = None
        for node in ast.walk(main_class):
            if isinstance(node, ast.FunctionDef) and node.name == "setup":
                main_setup = node
                break
        
        if main_setup:
            # Check embedding
            embed_parallel = False
            lm_head_parallel = False
            
            for node in ast.walk(main_setup):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Attribute):
                            if target.attr == 'embed_tokens':
                                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'ParallelEmbed':
                                    embed_parallel = True
                            elif target.attr == 'lm_head':
                                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'ParallelDense':
                                    lm_head_parallel = True
            
            print(f"  Embedding: {'✅ ParallelEmbed' if embed_parallel else '❌ NOT ParallelEmbed'}")
            print(f"  LM Head: {'✅ ParallelDense' if lm_head_parallel else '❌ NOT ParallelDense'}")
        else:
            print("  ❌ Could not find main model setup method")
            embed_parallel = False
            lm_head_parallel = False
    else:
        print("  ❌ Could not find Qwen25ForCausalLM class")
        embed_parallel = False
        lm_head_parallel = False
    
    print(f"\n--- FINAL RESULT ---")
    if all_parallel:
        print("✅ ALL ATTENTION PROJECTIONS USE ParallelDense")
        print("✅ TRUE FULL TENSOR PARALLELISM CONFIRMED")
    else:
        print("❌ NOT ALL ATTENTION PROJECTIONS USE ParallelDense")
        print("❌ FULL TENSOR PARALLELISM NOT ACHIEVED")
    
    if all_mlp_parallel:
        print("✅ ALL MLP PROJECTIONS USE ParallelDense")
    else:
        print("❌ NOT ALL MLP PROJECTIONS USE ParallelDense")
    
    print(f"\n=== VERIFICATION COMPLETE ===")
    
    return all_parallel and all_mlp_parallel

if __name__ == "__main__":
    verify_full_parallel() 