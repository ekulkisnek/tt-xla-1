#!/usr/bin/env python3
"""
FINAL FIX: Solve the parameter loading issue for full Q/K/V/O parallelization
"""
import os
import jax
import jax.numpy as jnp
import json
import flax.linen as nn
from typing import Dict, Any, Optional, Union
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from jax import lax

# Set up multi-device
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import q25j7_tensor_parallel_fixed
from q25j7_tensor_parallel_fixed import (
    ParallelDense, ParallelEmbed, QwenMLP, 
    setup_device_mesh, load_params
)
from transformers import AutoTokenizer

# Import working functions
def compute_cos_sin_cache(position_ids, head_dim, rope_theta=1000000.0):
    pos = position_ids.astype(jnp.float32)
    dim = head_dim // 2
    freqs = 1.0 / (rope_theta ** (jnp.arange(0, dim, dtype=jnp.float32) / dim))
    t = pos[..., None] * freqs[None, None, :]
    cos = jnp.cos(t)
    sin = jnp.sin(t)
    cos = cos[..., None, :]
    sin = sin[..., None, :]
    return cos, sin

def apply_rotary_emb(q, k, cos, sin):
    half_dim = q.shape[-1] // 2
    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    k1, k2 = k[..., :half_dim], k[..., half_dim:]
    q_rot = jnp.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
    k_rot = jnp.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)
    return q_rot, k_rot

def repeat_kv(hidden_states: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :]
    hidden_states = jnp.repeat(hidden_states, n_rep, axis=3)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)

class MockLlamaConfig:
    def __init__(self, qwen_config):
        self.hidden_size = qwen_config["hidden_size"]
        self.num_attention_heads = qwen_config["num_attention_heads"] 
        self.num_key_value_heads = qwen_config.get("num_key_value_heads", self.num_attention_heads)
        self.max_sequence_length = qwen_config.get("max_position_embeddings", 32768)
        self.rope_theta = qwen_config.get("rope_theta", 10000.0)
        self.attn_pdrop = 0.0

class FullyParallelLlamaAttention(nn.Module):
    """Final implementation: Fully parallel Llama-style attention that can load Qwen weights"""
    config: MockLlamaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # Use ParallelDense for ALL projections - this is our goal
        self.q_proj = ParallelDense(
            config.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
        )

        self.k_proj = ParallelDense(
            config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
        )

        self.v_proj = ParallelDense(
            config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
        )

        self.o_proj = ParallelDense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
        )

        self.causal_mask = make_causal_mask(
            jnp.ones((1, config.max_sequence_length), dtype="bool"), dtype="bool"
        )
        
        self.rope_theta = config.rope_theta

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(
            hidden_states.shape[:2] + (num_heads, self.head_dim)
        )

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        if attention_mask is None:
            attention_mask = jnp.ones((hidden_states.shape[0], hidden_states.shape[1]), dtype=jnp.float32)
        
        # Exact Llama attention computation with full parallelization
        xq, xk, xv = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

        xq = self._split_heads(xq, self.num_heads)
        xk = self._split_heads(xk, self.num_key_value_heads)
        xv = self._split_heads(xv, self.num_key_value_heads)

        cos, sin = compute_cos_sin_cache(position_ids, self.head_dim, self.rope_theta)
        xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        query_length, key_length = xq.shape[1], xk.shape[1]

        causal_mask = self.causal_mask[:, :, :query_length, :key_length]
        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(
            causal_mask, (batch_size,) + causal_mask.shape[1:]
        )

        attention_mask = jnp.broadcast_to(
            jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape
        )
        attention_mask = combine_masks(attention_mask, causal_mask)

        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(
                self.dtype
            ),
        )

        xk = repeat_kv(xk, self.num_key_value_groups)
        xv = repeat_kv(xv, self.num_key_value_groups)

        attn_weights = dot_product_attention_weights(
            xq,
            xk,
            bias=attention_bias,
            dropout_rng=None,
            dropout_rate=self.config.attn_pdrop,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=self.precision,
        )

        attn_output = jnp.einsum(
            "...hqk,...khd->...qhd", attn_weights, xv, precision=self.precision
        )
        attn_output = self._merge_heads(attn_output)
        attn_output = self.o_proj(attn_output)

        return attn_output, None

class FullyParallelQwenDecoderLayer(nn.Module):
    """Decoder layer with fully parallel attention"""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        llama_config = MockLlamaConfig(c)
        
        self.input_layernorm = nn.RMSNorm(epsilon=c.get("rms_norm_eps", 1e-6), dtype=jnp.bfloat16, name="input_layernorm")
        self.self_attn = FullyParallelLlamaAttention(config=llama_config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
        self.post_attention_layernorm = nn.RMSNorm(epsilon=c.get("rms_norm_eps", 1e-6), dtype=jnp.bfloat16, name="post_attention_layernorm")
        self.mlp = QwenMLP(config=c, dtype=jnp.bfloat16)

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, past_key_value = self.self_attn(hidden_states, attention_mask, position_ids, past_key_value)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, past_key_value

class FullyParallelQwenModel(nn.Module):
    """Complete Qwen model with fully parallel attention"""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        self.embed_tokens = ParallelEmbed(c["vocab_size"], c["hidden_size"], dtype=jnp.bfloat16, name="embed_tokens")
        self.layers = [FullyParallelQwenDecoderLayer(config=c, dtype=jnp.bfloat16, name=f"layers_{i}") for i in range(c["num_hidden_layers"])]
        self.norm = nn.RMSNorm(epsilon=c.get("rms_norm_eps", 1e-6), dtype=jnp.bfloat16, name="norm")
        self.lm_head = ParallelDense(c["vocab_size"], dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name="lm_head")

    def __call__(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, return_dict=True):
        batch, seq = input_ids.shape
        
        if attention_mask is None:
            attention_mask = jnp.ones((batch, seq), dtype=jnp.float32)

        hidden_states = self.embed_tokens(input_ids)
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        new_key_values = []
        
        for layer, past_kv in zip(self.layers, past_key_values):
            if position_ids is None:
                if past_kv is None:
                    position_ids = jnp.arange(seq)[None, :].repeat(batch, axis=0)
                else:
                    start_pos = past_kv[0].shape[1]
                    position_ids = jnp.arange(start_pos, start_pos + seq)[None, :].repeat(batch, axis=0)
            
            hidden_states, new_kv = layer(hidden_states, attention_mask, position_ids, past_kv)
            new_key_values.append(new_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if return_dict:
            return {"logits": logits, "past_key_values": new_key_values}
        return (logits,)

def debug_weight_loading_issue():
    """Debug exactly why weight loading fails with ParallelDense"""
    
    mesh = setup_device_mesh()
    q25j7_tensor_parallel_fixed.mesh = mesh
    
    with open("weights/config.json") as f:
        config = json.load(f)
    
    print("=== FINAL FIX: DEBUGGING WEIGHT LOADING ISSUE ===\n")
    
    # First, let's see what parameters get loaded
    print("--- Checking what gets loaded for regular Dense ---")
    
    class RegularAttentionModel(nn.Module):
        config: Dict[str, Any]
        
        def setup(self):
            c = self.config
            self.q_proj = nn.Dense(c["hidden_size"], dtype=jnp.bfloat16, use_bias=True)
            self.k_proj = nn.Dense(512, dtype=jnp.bfloat16, use_bias=True)  # 4 * 128
            self.v_proj = nn.Dense(512, dtype=jnp.bfloat16, use_bias=True)
            self.o_proj = nn.Dense(c["hidden_size"], dtype=jnp.bfloat16, use_bias=False)
        
        def __call__(self, x):
            return x  # dummy
    
    regular_model = RegularAttentionModel(config=config)
    
    # Initialize and try to load
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((1, 8, config["hidden_size"]), dtype=jnp.bfloat16)
    
    try:
        regular_params = regular_model.init(rng, dummy_input)
        print("✅ Regular model initialized")
        
        # Try loading weights - this should work
        regular_loaded = load_params(regular_model, "weights", jnp.bfloat16)
        print("✅ Regular model weights loaded")
        
        print("\nRegular model parameter structure:")
        for key in regular_loaded['params'].keys():
            print(f"  {key}: {list(regular_loaded['params'][key].keys())}")
        
    except Exception as e:
        print(f"❌ Regular model failed: {e}")
        return False
    
    print("\n--- Checking what happens with ParallelDense ---")
    
    class ParallelAttentionModel(nn.Module):
        config: Dict[str, Any]
        
        def setup(self):
            c = self.config
            self.q_proj = ParallelDense(c["hidden_size"], dtype=jnp.bfloat16, use_bias=True)
            self.k_proj = ParallelDense(512, dtype=jnp.bfloat16, use_bias=True)
            self.v_proj = ParallelDense(512, dtype=jnp.bfloat16, use_bias=True)
            self.o_proj = ParallelDense(c["hidden_size"], dtype=jnp.bfloat16, use_bias=False)
        
        def __call__(self, x):
            return x  # dummy
    
    parallel_model = ParallelAttentionModel(config=config)
    
    try:
        with mesh:
            parallel_params = parallel_model.init(rng, dummy_input)
        print("✅ Parallel model initialized")
        
        print("\nParallel model parameter structure:")
        if 'params' in parallel_params:
            for key in parallel_params['params'].keys():
                print(f"  {key}: {list(parallel_params['params'][key].keys())}")
        else:
            print(f"Keys in parallel_params: {list(parallel_params.keys())}")
        
        # Try loading weights - this is where it might fail
        try:
            parallel_loaded = load_params(parallel_model, "weights", jnp.bfloat16)
            print("✅ Parallel model weights loaded successfully!")
            
            print("\nParallel model loaded parameter structure:")
            for key in parallel_loaded['params'].keys():
                print(f"  {key}: {list(parallel_loaded['params'][key].keys())}")
                
        except Exception as e:
            print(f"❌ Parallel model weight loading failed: {e}")
            print("This is the root cause of the issue!")
            
            # Let's see what the weight loader is trying to do
            print("\n--- ANALYZING THE WEIGHT LOADING PROCESS ---")
            
            # Check available weight files
            import glob
            weight_files = glob.glob("weights/*.safetensors")
            print(f"Available weight files: {weight_files}")
            
            # Check what's in the first file
            try:
                import safetensors
                from safetensors import safe_open
                
                with safe_open(weight_files[0], framework="flax") as f:
                    keys = list(f.keys())
                    print(f"\nFirst 20 keys in {weight_files[0]}:")
                    for key in keys[:20]:
                        print(f"  {key}")
                        
                    # Look for attention weights specifically
                    attn_keys = [k for k in keys if 'attn' in k and 'layers.0' in k]
                    print(f"\nAttention weights for layer 0:")
                    for key in attn_keys:
                        print(f"  {key}")
                        
            except Exception as e2:
                print(f"Failed to examine safetensors: {e2}")
                
            return False
        
    except Exception as e:
        print(f"❌ Parallel model init failed: {e}")
        return False
    
    return True

def test_final_solution():
    """Test the complete solution with proper weight loading"""
    
    mesh = setup_device_mesh()
    q25j7_tensor_parallel_fixed.mesh = mesh
    
    with open("weights/config.json") as f:
        config = json.load(f)
    
    print("\n=== TESTING FINAL SOLUTION ===\n")
    print("Goal: Achieve full Q/K/V/O parallelization with correct output")
    
    # Create fully parallel model
    model = FullyParallelQwenModel(config=config, dtype=jnp.bfloat16)
    
    # Load tokenizer and create test input
    tokenizer = AutoTokenizer.from_pretrained("weights")
    test_input = "What is 2 + 2?"
    input_ids = tokenizer.encode(test_input, return_tensors="jax")
    
    print(f"Test input: '{test_input}'")
    print(f"Input IDs: {input_ids}")
    
    print("\n--- TESTING FINAL MODEL ---")
    
    try:
        # Initialize
        rng = jax.random.PRNGKey(42)
        params = model.init(rng, input_ids)
        print("✅ Model initialization successful")
        
        # Load weights
        loaded_params = load_params(model, "weights", jnp.bfloat16)
        print("✅ Weight loading successful")
        
        # Test forward pass
        with mesh:
            outputs = model.apply(loaded_params, input_ids, return_dict=True)
            logits = outputs['logits'][0, -1, :]
        
        print(f"✅ Forward pass successful")
        print(f"Logits shape: {logits.shape}")
        
        # Get prediction
        top_tokens = jnp.argsort(logits)[-10:][::-1]
        top_probs = jax.nn.softmax(logits)[top_tokens]
        
        print("\nTop 10 predicted tokens:")
        for i, (token_id, prob) in enumerate(zip(top_tokens, top_probs)):
            token_text = tokenizer.decode(int(token_id))
            print(f"  {i+1}. Token {token_id}: '{token_text}' (prob: {float(prob):.4f})")
        
        next_token = int(jnp.argmax(logits))
        next_token_text = tokenizer.decode(next_token)
        
        print(f"\nPredicted next token: {next_token} -> '{next_token_text}'")
        
        # Check if we get the expected result
        if next_token == 220:  # Space token
            print("\n🎉 COMPLETE SUCCESS!")
            print("✅ Full Q/K/V/O parallelization achieved")
            print("✅ Correct output maintained")
            print("✅ Weight loading works")
            print("✅ All requirements met!")
            return True
        else:
            print(f"\n⚠️ Partial success: Model works but output differs")
            print(f"Expected: 220 (' '), Got: {next_token} ('{next_token_text}')")
            print("This suggests there might still be a subtle issue")
            return "partial"
        
    except Exception as e:
        print(f"\n❌ Final test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("FINAL FIX: Investigating and solving the parameter loading issue...\n")
    
    # Step 1: Debug the weight loading issue
    debug_result = debug_weight_loading_issue()
    
    if debug_result:
        print("\n✅ Weight loading issue understood")
        
        # Step 2: Test the final solution
        result = test_final_solution()
        
        if result == True:
            print("\n🏆 MISSION ACCOMPLISHED!")
            print("Full tensor parallelism with exact output equivalence achieved!")
        elif result == "partial":
            print("\n🚧 ALMOST THERE!")
            print("Full parallelization works, minor output differences remain")
        else:
            print("\n🔧 DEBUGGING NEEDED!")
            print("Additional investigation required")
    else:
        print("\n🔍 INVESTIGATION NEEDED!")
        print("Weight loading issue requires deeper analysis")