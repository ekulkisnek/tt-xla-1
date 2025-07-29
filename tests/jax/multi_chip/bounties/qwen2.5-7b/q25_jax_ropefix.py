import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer
from safetensors.numpy import load_file
import glob

# Configuration
model_path = "weights"
dtype = jnp.float16

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Model config (hardcoded from Qwen2.5-7B-Instruct)
class Config:
    hidden_size = 3584
    num_attention_heads = 28
    num_key_value_heads = 4
    head_dim = 128
    num_hidden_layers = 28
    intermediate_size = 18944
    max_position_embeddings = 32768
    rms_norm_eps = 1e-6
    rope_theta = 10000.0

config = Config()

# Load weights with correct transposition
def load_weights():
    weight_files = glob.glob(f"{model_path}/model-*.safetensors")
    params = {}
    for file in weight_files:
        weights = load_file(file)
        for k, v in weights.items():
            if "proj" in k and "weight" in k:
                params[k] = v.T.astype(np.float16)
            else:
                params[k] = v.astype(np.float16)
    return params

params = load_weights()

# RoPE implementation
def compute_rotary_embeddings(max_len, head_dim, theta=config.rope_theta):
    position = jnp.arange(max_len, dtype=jnp.float32)
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = position[:, None] * freqs[None, :]
    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)
    return cos, sin

def apply_rotary_emb(x, cos, sin):
    # x: (batch, seq, heads, head_dim)
    batch, seq, heads, head_dim = x.shape
    half_dim = head_dim // 2
    x1 = x[..., :half_dim]  # First half
    x2 = x[..., half_dim:]  # Second half
    # Fix broadcasting: cos/sin shape (1, seq, 1, 64)
    cos = cos[..., None, :]
    sin = sin[..., None, :]
    # Rotate
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x2 * cos + x1 * sin
    # Interleave instead of concatenate
    x_rot = jnp.zeros_like(x)
    x_rot = x_rot.at[..., 0::2].set(x1_rot)
    x_rot = x_rot.at[..., 1::2].set(x2_rot)
    return x_rot

# RMSNorm
def rms_norm(x, weight, eps=config.rms_norm_eps):
    mean_sq = jnp.mean(x ** 2, axis=-1, keepdims=True)
    return x * weight / jnp.sqrt(mean_sq + eps)

# Attention mechanism with GQA
def attention(q, k, v, attention_mask):
    # q, k, v: (batch, seq, heads, head_dim)
    batch, seq, heads, head_dim = q.shape
    repeat_factor = config.num_attention_heads // config.num_key_value_heads
    # Transpose to (batch, heads, seq, head_dim)
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    # Repeat k/v for GQA
    k = jnp.repeat(k, repeat_factor, axis=1)
    v = jnp.repeat(v, repeat_factor, axis=1)
    # Compute attention scores
    scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) / jnp.sqrt(head_dim)
    scores = scores + (1.0 - attention_mask) * -1e9
    attn = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
    # Transpose back to (batch, seq, heads, head_dim)
    out = out.transpose(0, 2, 1, 3)
    return out

# Model forward pass
def model_forward(input_ids, position_ids, attention_mask, params):
    hidden_states = params["model.embed_tokens.weight"][input_ids]
    
    cos, sin = compute_rotary_embeddings(config.max_position_embeddings, config.head_dim)
    cos = cos[position_ids]
    sin = sin[position_ids]
    
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        hidden_states = rms_norm(hidden_states, params[f"{prefix}.input_layernorm.weight"])
        
        q = hidden_states @ params[f"{prefix}.self_attn.q_proj.weight"]
        k = hidden_states @ params[f"{prefix}.self_attn.k_proj.weight"]
        v = hidden_states @ params[f"{prefix}.self_attn.v_proj.weight"]
        
        q = q.reshape(1, -1, config.num_attention_heads, config.head_dim)
        k = k.reshape(1, -1, config.num_key_value_heads, config.head_dim)
        v = v.reshape(1, -1, config.num_key_value_heads, config.head_dim)
        
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        attn_output = attention(q, k, v, attention_mask)
        attn_output = attn_output.reshape(-1, config.hidden_size)
        hidden_states = hidden_states + (attn_output @ params[f"{prefix}.self_attn.o_proj.weight"])
        
        hidden_states = rms_norm(hidden_states, params[f"{prefix}.post_attention_layernorm.weight"])
        gate_proj = hidden_states @ params[f"{prefix}.mlp.gate_proj.weight"]
        up_proj = hidden_states @ params[f"{prefix}.mlp.up_proj.weight"]
        hidden_states = (gate_proj * jax.nn.silu(up_proj)) @ params[f"{prefix}.mlp.down_proj.weight"]
    
    hidden_states = rms_norm(hidden_states, params["model.norm.weight"])
    logits = hidden_states @ params["lm_head.weight"].T
    return logits

# Generation loop
def generate(prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    
    position_ids = np.arange(seq_len)[None, :]
    attention_mask = np.ones((1, 1, 1, seq_len), dtype=np.int32)
    
    for _ in range(max_new_tokens):
        logits = model_forward(input_ids, position_ids, attention_mask, params)
        next_token = jnp.argmax(logits[:, -1, :], axis=-1)
        input_ids = np.concatenate([input_ids, next_token[:, None]], axis=1)
        position_ids = np.concatenate([position_ids, position_ids[:, -1:] + 1], axis=1)
        attention_mask = np.ones((1, 1, 1, input_ids.shape[1]), dtype=np.int32)
        
        if next_token[0] == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# Test with GSM8K-style question
prompt = """System: You are a helpful AI assistant. Please answer the following math question clearly and concisely.\n\nQuestion: A store has 5 apples and buys 3 more. How many apples does it have now?"""
response = generate(prompt)
print(response) 