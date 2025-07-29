import jax
import jax.numpy as jnp
from jax import jit, lax
import numpy as np
from transformers import AutoTokenizer
from safetensors import safe_open
import json
from typing import Dict, Tuple, Optional
import os
import gc

# Config class to match PyTorch's structure
class Qwen25Config:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.hidden_size = config['hidden_size']
        self.num_attention_heads = config['num_attention_heads']
        self.num_key_value_heads = config['num_key_value_heads']
        self.num_hidden_layers = config['num_hidden_layers']
        self.rope_theta = config.get('rope_theta', 1e6)
        self.max_position_embeddings = config['max_position_embeddings']
        self.vocab_size = config['vocab_size']
        self.intermediate_size = config['intermediate_size']
        self.rms_norm_eps = config.get('rms_norm_eps', 1e-6)

# RMSNorm implementation
def rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    x = x * jax.lax.rsqrt(variance + eps)
    return x * weight

# RoPE implementation
def apply_rotary_pos_emb(q: jnp.ndarray, k: jnp.ndarray, positions: jnp.ndarray, theta: float = 1e6) -> Tuple[jnp.ndarray, jnp.ndarray]:
    dim = q.shape[-1]
    angles = 1.0 / (theta ** (jnp.arange(0, dim, 2) / dim))
    angles = positions[..., None] * angles[None, :]
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    
    # Reshape cos and sin to match q and k dimensions
    cos = cos[..., None, :]  # Add head dimension
    sin = sin[..., None, :]  # Add head dimension
    
    q_even, q_odd = q[..., 0::2], q[..., 1::2]
    k_even, k_odd = k[..., 0::2], k[..., 1::2]
    
    q_rot = jnp.concatenate([q_even * cos - q_odd * sin, q_even * sin + q_odd * cos], axis=-1)
    k_rot = jnp.concatenate([k_even * cos - k_odd * sin, k_even * sin + k_odd * cos], axis=-1)
    return q_rot, k_rot

# Attention mechanism with GQA
class QwenAttention:
    def __init__(self, config: Qwen25Config, params: Dict):
        self.config = config
        self.q_proj = params['q_proj']
        self.k_proj = params['k_proj']
        self.v_proj = params['v_proj']
        self.o_proj = params['o_proj']
        self.head_dim = config.hidden_size // config.num_attention_heads

    def __call__(self, hidden_states: jnp.ndarray, attention_mask: jnp.ndarray, position_ids: jnp.ndarray,
                 cache: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        batch_size, seq_len = hidden_states.shape[:2]
        
        print(f"hidden_states shape: {hidden_states.shape}")
        print(f"q_proj shape: {self.q_proj.shape}")
        print(f"k_proj shape: {self.k_proj.shape}")
        print(f"v_proj shape: {self.v_proj.shape}")
        print(f"head_dim: {self.head_dim}")
        print(f"num_attention_heads: {self.config.num_attention_heads}")
        print(f"num_key_value_heads: {self.config.num_key_value_heads}")
        
        q = jnp.dot(hidden_states, self.q_proj).reshape(batch_size, seq_len, self.config.num_attention_heads, self.head_dim)
        k = jnp.dot(hidden_states, self.k_proj.T).reshape(batch_size, seq_len, self.config.num_key_value_heads, self.head_dim)
        v = jnp.dot(hidden_states, self.v_proj.T).reshape(batch_size, seq_len, self.config.num_key_value_heads, self.head_dim)

        q, k = apply_rotary_pos_emb(q, k, position_ids, self.config.rope_theta)

        if cache is not None:
            past_k, past_v = cache
            k = jnp.concatenate([past_k, k], axis=1)
            v = jnp.concatenate([past_v, v], axis=1)

        q = q.transpose(0, 2, 1, 3)  # (batch, heads, seq, dim)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(self.head_dim)
        scores = scores + attention_mask
        probs = jax.nn.softmax(scores, axis=-1)
        probs = jnp.clip(probs, 1e-9, 1 - 1e-9)

        attn_output = jnp.matmul(probs, v).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        output = jnp.dot(attn_output, self.o_proj.T)

        return output, (k, v)

# MLP implementation
class QwenMLP:
    def __init__(self, config: Qwen25Config, params: Dict):
        self.gate_proj = params['gate_proj']
        self.up_proj = params['up_proj']
        self.down_proj = params['down_proj']

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate = jax.nn.silu(jnp.dot(x, self.gate_proj.T))
        up = jnp.dot(x, self.up_proj.T)
        return jnp.dot(gate * up, self.down_proj.T)

# Decoder layer
class QwenDecoderLayer:
    def __init__(self, config: Qwen25Config, params: Dict):
        self.self_attn = QwenAttention(config, params['self_attn'])
        self.mlp = QwenMLP(config, params['mlp'])
        self.input_layernorm = params['input_layernorm']
        self.post_attention_layernorm = params['post_attention_layernorm']
        self.config = config

    def __call__(self, hidden_states: jnp.ndarray, attention_mask: jnp.ndarray, position_ids: jnp.ndarray,
                 cache: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        residual = hidden_states
        hidden_states = rms_norm(hidden_states, self.input_layernorm, self.config.rms_norm_eps)
        attn_output, new_cache = self.self_attn(hidden_states, attention_mask, position_ids, cache)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = rms_norm(hidden_states, self.post_attention_layernorm, self.config.rms_norm_eps)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, new_cache

# Main model class
class Qwen25ForCausalLM:
    def __init__(self, config: Qwen25Config, params: Dict):
        self.config = config
        self.embed_tokens = params['embed_tokens']
        self.layers = [QwenDecoderLayer(config, params['layers'][i]) for i in range(config.num_hidden_layers)]
        self.norm = params['norm']
        self.lm_head = self.embed_tokens  # Weight tying

    def __call__(self, input_ids: jnp.ndarray, attention_mask: jnp.ndarray, position_ids: jnp.ndarray,
                 cache: Optional[list] = None) -> Tuple[jnp.ndarray, list]:
        hidden_states = self.embed_tokens[input_ids]
        new_cache = []

        for i, layer in enumerate(self.layers):
            cache_layer = cache[i] if cache is not None else None
            hidden_states, layer_cache = layer(hidden_states, attention_mask, position_ids, cache_layer)
            new_cache.append(layer_cache)

        hidden_states = rms_norm(hidden_states, self.norm, self.config.rms_norm_eps)
        logits = jnp.dot(hidden_states, self.lm_head.T)
        return logits, new_cache

# Sampling function
@jit
def sample(logits: jnp.ndarray, temperature: float = 0.7, top_k: int = 40, top_p: float = 0.9,
           repetition_penalty: float = 1.1, past_ids: jnp.ndarray = None, seed: int = 42) -> jnp.ndarray:
    logits = jnp.clip(logits, -20, 20)
    if past_ids is not None:
        mask = jnp.isin(jnp.arange(logits.shape[-1]), past_ids)
        logits = jnp.where(mask, logits / repetition_penalty, logits)

    logits = logits / temperature
    if top_k > 0:
        indices_to_remove = logits < jnp.sort(logits, axis=-1)[..., -top_k]
        logits = jnp.where(indices_to_remove, -jnp.inf, logits)
    if top_p < 1.0:
        sorted_logits = jnp.sort(logits, axis=-1)[..., ::-1]
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        indices_to_remove = sorted_indices_to_remove[..., 1:] | (cumulative_probs[..., :-1] > top_p)
        logits = jnp.where(indices_to_remove[jnp.argsort(jnp.argsort(logits, axis=-1), axis=-1)], -jnp.inf, logits)

    key = jax.random.PRNGKey(seed)
    return jax.random.categorical(key, logits)

# Load model and weights
def load_model(model_path: str) -> Tuple[Qwen25ForCausalLM, AutoTokenizer]:
    print(f"Loading config from {model_path}/config.json")
    config = Qwen25Config(f"{model_path}/config.json")
    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading model weights...")
    params = {}
    
    # Load from multiple safetensors files
    safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
    safetensors_files.sort()
    
    # Load embed_tokens and norm from all files
    for file in safetensors_files:
        print(f"Loading embed_tokens and norm from {file}")
        with safe_open(f"{model_path}/{file}", framework="pt") as st:
            if 'model.embed_tokens.weight' in st.keys() and 'embed_tokens' not in params:
                tensor = st.get_tensor('model.embed_tokens.weight')
                if hasattr(tensor, 'detach'):
                    # Handle PyTorch tensor
                    params['embed_tokens'] = tensor.detach().cpu().float().numpy().astype(jnp.bfloat16)
                else:
                    # Handle numpy array
                    params['embed_tokens'] = tensor.astype(np.float32).astype(jnp.bfloat16)
            if 'model.norm.weight' in st.keys() and 'norm' not in params:
                tensor = st.get_tensor('model.norm.weight')
                if hasattr(tensor, 'detach'):
                    # Handle PyTorch tensor
                    params['norm'] = tensor.detach().cpu().float().numpy().astype(jnp.bfloat16)
                else:
                    # Handle numpy array
                    params['norm'] = tensor.astype(np.float32).astype(jnp.bfloat16)
    
    params['layers'] = []
    for i in range(config.num_hidden_layers):
        layer_params = {
            'self_attn': {},
            'mlp': {},
        }
        
        # Load layer weights from all files
        for file in safetensors_files:
            print(f"Loading layer {i} from {file}")
            with safe_open(f"{model_path}/{file}", framework="pt") as st:
                # Attention weights
                for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    key = f'model.layers.{i}.self_attn.{proj}.weight'
                    if key in st.keys() and proj not in layer_params['self_attn']:
                        tensor = st.get_tensor(key)
                        if hasattr(tensor, 'detach'):
                            layer_params['self_attn'][proj] = tensor.detach().cpu().float().numpy().astype(jnp.bfloat16)
                        else:
                            layer_params['self_attn'][proj] = tensor.astype(np.float32).astype(jnp.bfloat16)
                
                # MLP weights
                for proj in ['gate_proj', 'up_proj', 'down_proj']:
                    key = f'model.layers.{i}.mlp.{proj}.weight'
                    if key in st.keys() and proj not in layer_params['mlp']:
                        tensor = st.get_tensor(key)
                        if hasattr(tensor, 'detach'):
                            layer_params['mlp'][proj] = tensor.detach().cpu().float().numpy().astype(jnp.bfloat16)
                        else:
                            layer_params['mlp'][proj] = tensor.astype(np.float32).astype(jnp.bfloat16)
                
                # Layer norm weights
                for norm in ['input_layernorm', 'post_attention_layernorm']:
                    key = f'model.layers.{i}.{norm}.weight'
                    if key in st.keys() and norm not in layer_params:
                        tensor = st.get_tensor(key)
                        if hasattr(tensor, 'detach'):
                            layer_params[norm] = tensor.detach().cpu().float().numpy().astype(jnp.bfloat16)
                        else:
                            layer_params[norm] = tensor.astype(np.float32).astype(jnp.bfloat16)
        
        params['layers'].append(layer_params)
        gc.collect()

    # Validate weight tying
    print("Validating weight tying...")
    assert jnp.allclose(params['embed_tokens'], params['embed_tokens']), "Weight tying validation failed"
    print("Model loaded successfully!")
    return Qwen25ForCausalLM(config, params), tokenizer

# Generation function
def generate(model: Qwen25ForCausalLM, tokenizer: AutoTokenizer, prompt: str, max_length: int = 512) -> str:
    system_prompt = "You are a highly capable math assistant. Solve the problem step-by-step and provide a clear, concise answer."
    input_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>"
    inputs = tokenizer(input_text, return_tensors="np", padding=False)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    batch_size, seq_len = input_ids.shape
    position_ids = jnp.arange(seq_len)[None, :]
    cache = None

    generated_ids = input_ids
    for step in range(max_length - seq_len):
        print(f"Generating step {step + 1}...", end=" ", flush=True)
        logits, cache = model(generated_ids, attention_mask, position_ids, cache)
        next_token = sample(logits[:, -1, :], past_ids=generated_ids)
        generated_ids = jnp.concatenate([generated_ids, next_token[:, None]], axis=1)
        attention_mask = jnp.ones_like(generated_ids)
        position_ids = jnp.arange(generated_ids.shape[1])[None, :]

        # Decode and print the token
        token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
        print(f"'{token_text}'")

        if next_token.item() in [151643, tokenizer.convert_tokens_to_ids("<|im_end|>")] or generated_ids.shape[1] >= max_length:
            break

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    model_path = "weights"  # Use the weights directory in current path
    print("Loading Qwen 2.5-7B model...")
    model, tokenizer = load_model(model_path)
    
    prompt = "Janet's dogs eat 2 pounds of dog food each day. If Janet buys a 20-pound bag of dog food, how many days will it last?"
    print(f"\nGenerating response for: {prompt}")
    print("=" * 80)
    
    output = generate(model, tokenizer, prompt, max_length=200)
    print("\n" + "=" * 80)
    print("Final output:")
    print(output) 