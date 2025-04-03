import jax
import jax.numpy as jnp
import optax
import numpy as np
from tqdm import tqdm
from model import CodeGPT
from preprocess import create_dataset
from flax.training import train_state
from typing import Tuple, Dict
import os

def create_train_state(rng, model, learning_rate):
    """Creates initial training state."""
    params = model.init(rng, jnp.ones((1, 1), dtype=jnp.int32))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

@jax.jit
def train_step(state, batch):
    """Performs a single training step."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['input_ids'], training=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits[:, :-1], batch['input_ids'][:, 1:]
        ).mean()
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def train(
    num_epochs: int = 10,
    batch_size: int = 32,
    max_len: int = 512,
    learning_rate: float = 1e-4,
    num_layers: int = 6,
    num_heads: int = 8,
    head_dim: int = 64,
    mlp_dim: int = 2048,
    save_dir: str = 'checkpoints',
    tokenizer_path: str = 'tokenizer',
    dataset_path: str = "filtered_python_dataset",
    streaming: bool = False
):
    """Main training loop."""
    # Create dataset
    dataset = create_dataset(
        tokenizer_path=tokenizer_path,
        max_length=max_len,
        dataset_path=dataset_path,
        streaming=streaming
    )
    
    # Create model
    model = CodeGPT(
        vocab_size=dataset.vocab_size,
        max_len=max_len,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        mlp_dim=mlp_dim
    )
    
    # Initialize training state
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, model, learning_rate)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        # Get batch
        batch = dataset.get_batch(batch_size)
        
        # Training step
        state, loss = train_step(state, batch)
        
        # Print progress
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_{epoch + 1}')
            with open(checkpoint_path, 'wb') as f:
                f.write(jax.serialization.to_bytes(state.params))
            
            # Save vocabulary
            dataset.save_vocab(os.path.join(save_dir, f'vocab_{epoch + 1}.json'))

if __name__ == '__main__':
    train() 