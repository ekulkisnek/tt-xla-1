import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from transformers import AutoTokenizer
from safetensors.numpy import load_file
import glob
import os

# Configuration
model_path = "weights"  # Use local weights directory
dtype = jnp.float16  # Match PyTorch's torch.float16

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Model config (hardcoded from Qwen2.5-7B-Instruct)
class Config:
    hidden_size = 3584
    num_attention_heads = 28
    num_key_value_heads = 4
    head_dim = 128  # hidden_size // num_attention_heads
    num_hidden_layers = 28
    intermediate_size = 18944
    max_position_embeddings = 32768
    rms_norm_eps = 1e-6
    rope_theta = 10000.0

config = Config()

# Load weights
def load_weights():
    print("Loading weights...")
    weight_files = glob.glob(f"{model_path}/model-*.safetensors")
    weight_files.sort()  # Ensure consistent ordering
    params = {}
    for file in weight_files:
        print(f"Loading {file}")
        weights = load_file(file)
        for k, v in weights.items():
            if "norm" in k and k not in params:  # Ensure norm weights are loaded
                params[k] = v.astype(np.float16)
            elif k not in params:
                # Transpose weight matrices for JAX compatibility
                if "weight" in k and ("proj" in k or "lm_head" in k):
                    params[k] = v.T.astype(np.float16)
                elif "embed_tokens.weight" in k:
                    # Embedding weights should not be transposed
                    params[k] = v.astype(np.float16)
                else:
                    params[k] = v.astype(np.float16)
    print(f"Loaded {len(params)} parameters")
    return params

params = load_weights()

# RoPE implementation
def compute_cos_sin_cache(max_len, head_dim, theta=config.rope_theta):
    position = jnp.arange(max_len, dtype=jnp.float32)
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = position[:, None] * freqs[None, :]
    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)
    return cos, sin

def apply_rotary_emb(q, k, cos, sin, position_ids):
    # Get the position indices for the current sequence
    pos_indices = position_ids[0]  # Remove batch dimension
    
    # Select the relevant cos/sin values for current positions
    cos_pos = cos[pos_indices]
    sin_pos = sin[pos_indices]
    
    # Reshape to match q and k dimensions
    cos_pos = cos_pos[None, None, :, :]  # Add batch and head dimensions
    sin_pos = sin_pos[None, None, :, :]  # Add batch and head dimensions
    
    # Apply RoPE
    q_even = q[..., ::2]
    q_odd = q[..., 1::2]
    q_rot = jnp.concatenate([-q_odd, q_even], axis=-1)
    q_rot = q_rot * sin_pos + q * cos_pos
    
    k_even = k[..., ::2]
    k_odd = k[..., 1::2]
    k_rot = jnp.concatenate([-k_odd, k_even], axis=-1)
    k_rot = k_rot * sin_pos + k * cos_pos
    
    return q_rot, k_rot

# RMSNorm
def rms_norm(x, weight, eps=config.rms_norm_eps):
    mean_sq = jnp.mean(x ** 2, axis=-1, keepdims=True)
    return x * weight / jnp.sqrt(mean_sq + eps)

# Attention mechanism
def attention(q, k, v, attention_mask):
    batch, heads, seq, head_dim = q.shape
    k = jnp.repeat(k, config.num_attention_heads // config.num_key_value_heads, axis=1)  # GQA fix
    v = jnp.repeat(v, config.num_attention_heads // config.num_key_value_heads, axis=1)  # GQA fix
    k = k.transpose(0, 1, 3, 2)  # (batch, heads, head_dim, seq)
    scores = q @ k / jnp.sqrt(head_dim)
    scores = scores + (1.0 - attention_mask) * -1e9
    attn = jax.nn.softmax(scores, axis=-1)
    return attn @ v

# Sampling function (JIT-compiled)
@jit
def sample(logits, key, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.1):
    logits = logits / temperature
    # Apply repetition penalty (simplified, assumes past tokens not tracked here)
    logits = logits / repetition_penalty
    # Top-k and top-p filtering
    top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
    mask = jax.random.categorical(key, top_k_logits) < (top_p * top_k_logits.sum())
    return top_k_indices[mask]

# Model forward pass
def model_forward(input_ids, position_ids, attention_mask, params, cos, sin):
    hidden_states = params["model.embed_tokens.weight"][input_ids]
    
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        # Layer norm
        hidden_states = rms_norm(hidden_states, params[f"{prefix}.input_layernorm.weight"])
        
        # Attention
        q = hidden_states @ params[f"{prefix}.self_attn.q_proj.weight"]
        k = hidden_states @ params[f"{prefix}.self_attn.k_proj.weight"]
        v = hidden_states @ params[f"{prefix}.self_attn.v_proj.weight"]
        
        batch_size, seq_len = hidden_states.shape[:2]
        q = q.reshape(batch_size, seq_len, config.num_attention_heads, config.head_dim)
        k = k.reshape(batch_size, seq_len, config.num_key_value_heads, config.head_dim)
        v = v.reshape(batch_size, seq_len, config.num_key_value_heads, config.head_dim)
        
        q, k = apply_rotary_emb(q, k, cos, sin, position_ids)
        attn_output = attention(q, k, v, attention_mask)
        attn_output = attn_output.reshape(batch_size, seq_len, config.hidden_size)
        hidden_states = hidden_states + (attn_output @ params[f"{prefix}.self_attn.o_proj.weight"])
        
        # MLP
        hidden_states = rms_norm(hidden_states, params[f"{prefix}.post_attention_layernorm.weight"])
        gate_proj = hidden_states @ params[f"{prefix}.mlp.gate_proj.weight"]
        up_proj = hidden_states @ params[f"{prefix}.mlp.up_proj.weight"]
        hidden_states = (gate_proj * jax.nn.silu(up_proj)) @ params[f"{prefix}.mlp.down_proj.weight"]
    
    hidden_states = rms_norm(hidden_states, params["model.norm.weight"])
    logits = hidden_states @ params["lm_head.weight"].T
    return logits

# Generation loop
def generate(prompt, max_new_tokens=256):
    print(f"Generating response for: {prompt[:100]}...")
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    
    cos, sin = compute_cos_sin_cache(config.max_position_embeddings, config.head_dim)
    position_ids = np.arange(seq_len)[None, :]
    attention_mask = np.ones((1, 1, 1, seq_len), dtype=np.int32)
    
    key = jax.random.PRNGKey(0)
    generated_ids = input_ids
    
    print("Starting generation...")
    for step in range(max_new_tokens):
        print(f"Step {step + 1}/{max_new_tokens}", end=" ", flush=True)
        logits = model_forward(generated_ids, position_ids, attention_mask, params, cos, sin)
        next_token = sample(logits[:, -1, :], key)
        generated_ids = np.concatenate([generated_ids, next_token[:, None]], axis=1)
        position_ids = np.array([[position_ids[0, -1] + 1]])
        attention_mask = np.ones((1, 1, 1, 1), dtype=np.int32)  # Update for new token
        key = jax.random.split(key)[0]
        
        # Decode and print token
        token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
        print(f"'{token_text}'")
        
        if next_token[0] == tokenizer.eos_token_id:
            print("EOS token reached")
            break
    
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Test with GSM8K-style question
prompt = """System: You are a helpful AI assistant. Please answer the following math question clearly and concisely.

Question: Janet's dogs eat 2 pounds of dog food each day. If Janet buys a 20-pound bag of dog food, how many days will it last?"""

print("=" * 80)
response = generate(prompt, max_new_tokens=100)
print("=" * 80)
print("Final response:")
print(response) 