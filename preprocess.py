import os
import json
from typing import List, Dict, Tuple
import numpy as np
import jax.numpy as jnp
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset, Dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

class CodeDataset:
    def __init__(
        self,
        tokenizer_path: str = None,
        max_length: int = 512,
        dataset_path: str = "filtered_python_dataset",
        split: str = "train",
        cache_dir: str = "data_cache",
        streaming: bool = False
    ):
        self.max_length = max_length
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize tokenizer
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        else:
            # Create a basic tokenizer for code
            self.tokenizer = self._create_code_tokenizer()
            if tokenizer_path:
                self.tokenizer.save_pretrained(tokenizer_path)
        
        # Load dataset
        if os.path.exists(dataset_path):
            self.dataset = Dataset.load_from_disk(dataset_path)
        else:
            # Load dataset in streaming mode
            streamed_data = load_dataset(
                "bigcode/the-stack",
                data_dir="data/python",
                split=split,
                streaming=True
            )
            
            # Take first 10,000 samples and convert to a standard Dataset
            subset = list(streamed_data.take(10_000))
            self.dataset = Dataset.from_list(subset)
            
            # Save locally
            self.dataset.save_to_disk(dataset_path)
            print("Dataset saved successfully!")
        
        # Create vocabulary
        self.vocab_size = len(self.tokenizer)
        
    def _create_code_tokenizer(self) -> PreTrainedTokenizerFast:
        """Create a basic tokenizer for code."""
        # Initialize tokenizer
        tokenizer = Tokenizer(models.BPE())
        
        # Configure pre-tokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        
        # Configure post-processor
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        
        # Train tokenizer on a sample of code
        trainer = trainers.BpeTrainer(
            vocab_size=50000,
            min_frequency=2,
            special_tokens=["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
        )
        
        # Train on the dataset
        def batch_iterator():
            for i in range(0, len(self.dataset), 1000):
                yield self.dataset[i:i+1000]["content"]
        
        tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
        
        # Return as PreTrainedTokenizerFast
        return PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    
    def _preprocess_code(self, code: str) -> str:
        """Preprocess code before tokenization."""
        # Remove comments
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            if not line.strip().startswith(('#', '//', '/*', '*/')):
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)
    
    def tokenize_batch(self, batch: Dict) -> Dict:
        """Tokenize a batch of code samples."""
        # Preprocess code
        processed_code = [self._preprocess_code(code) for code in batch['content']]
        
        # Tokenize
        tokenized = self.tokenizer(
            processed_code,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        }
    
    def get_batch(self, batch_size: int) -> Dict:
        """Get a batch of tokenized code samples."""
        # Sample from dataset
        batch = self.dataset.shuffle().select(range(batch_size))
        
        # Tokenize batch
        tokenized_batch = self.tokenize_batch(batch)
        
        return {
            'input_ids': jnp.array(tokenized_batch['input_ids']),
            'attention_mask': jnp.array(tokenized_batch['attention_mask'])
        }
    
    def save_vocab(self, path: str):
        """Save vocabulary to file."""
        vocab = self.tokenizer.get_vocab()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    def load_vocab(self, path: str):
        """Load vocabulary from file."""
        with open(path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        self.tokenizer.add_tokens(list(vocab.keys()))

def create_dataset(
    tokenizer_path: str = None,
    max_length: int = 512,
    dataset_path: str = "filtered_python_dataset",
    split: str = "train",
    cache_dir: str = "data_cache",
    streaming: bool = False
) -> CodeDataset:
    """Create and return a CodeDataset instance."""
    return CodeDataset(
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        dataset_path=dataset_path,
        split=split,
        cache_dir=cache_dir,
        streaming=streaming
    )

if __name__ == '__main__':
    # Example usage
    dataset = create_dataset(
        tokenizer_path="tokenizer",
        max_length=512,
        dataset_path="filtered_python_dataset",
        split="train"
    )
    
    # Get a batch of data
    batch = dataset.get_batch(batch_size=32)
    print(f"Batch shape: {batch['input_ids'].shape}")
    print(f"Vocabulary size: {dataset.vocab_size}")
    
    # Print a sample
    print("\nSample code from dataset:")
    print(dataset.dataset[0]["content"]) 