import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, training: bool = False) -> jnp.ndarray:
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations for Q, K, V
        q = nn.Dense(self.num_heads * self.head_dim)(x)
        k = nn.Dense(self.num_heads * self.head_dim)(x)
        v = nn.Dense(self.num_heads * self.head_dim)(x)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Scaled dot-product attention
        attention = jnp.einsum('bqhd,bkhd->bhqk', q, k) / jnp.sqrt(self.head_dim)
        
        if mask is not None:
            attention = jnp.where(mask == 0, -1e9, attention)
        
        attention = nn.softmax(attention, axis=-1)
        attention = nn.Dropout(rate=self.dropout_rate)(attention, deterministic=not training)
        
        # Apply attention to values
        out = jnp.einsum('bhqk,bkhd->bqhd', attention, v)
        out = out.reshape(batch_size, seq_len, -1)
        
        # Final linear transformation
        out = nn.Dense(x.shape[-1])(out)
        return out

class TransformerBlock(nn.Module):
    num_heads: int
    head_dim: int
    mlp_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, training: bool = False) -> jnp.ndarray:
        # Self-attention
        attn_out = MultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate
        )(x, mask, training)
        
        # Add & Norm
        x = x + nn.Dropout(rate=self.dropout_rate)(attn_out, deterministic=not training)
        x = nn.LayerNorm()(x)
        
        # Feed-forward
        ff = nn.Sequential([
            nn.Dense(self.mlp_dim),
            nn.gelu,
            nn.Dropout(rate=self.dropout_rate),
            nn.Dense(x.shape[-1])
        ])(x)
        
        # Add & Norm
        x = x + nn.Dropout(rate=self.dropout_rate)(ff, deterministic=not training)
        x = nn.LayerNorm()(x)
        
        return x

class CodeGPT(nn.Module):
    vocab_size: int
    max_len: int
    num_layers: int
    num_heads: int
    head_dim: int
    mlp_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        # Token embeddings
        x = nn.Embed(self.vocab_size, self.head_dim * self.num_heads)(x)
        
        # Position embeddings
        pos = jnp.arange(self.max_len)[None, :]
        pos_emb = nn.Embed(self.max_len, self.head_dim * self.num_heads)(pos)
        x = x + pos_emb
        
        # Create attention mask
        mask = jnp.triu(jnp.ones((x.shape[1], x.shape[1])), k=1)
        mask = mask == 0
        
        # Transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate
            )(x, mask, training)
        
        # Final layer norm
        x = nn.LayerNorm()(x)
        
        # Output projection
        x = nn.Dense(self.vocab_size)(x)
        
        return x 