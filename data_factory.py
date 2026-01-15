import torch
from torch.utils.data import Dataset, DataLoader
import os
import requests
from typing import Optional, Iterator

class TextFileDataset(Dataset):
    """ Simple dataset for plain text files (char-level or pre-tokenized) """
    def __init__(self, path, block_size, tokenizer):
        self.block_size = block_size
        self.tokenizer = tokenizer
        
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

class HFStreamDataset(torch.utils.data.IterableDataset):
    """ 
    Wrapper for HuggingFace Datasets Streaming.
    Useful for massive datasets like FineWeb-Edu without local download.
    """
    def __init__(self, dataset_name, split, block_size, tokenizer, max_steps=None):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.split = split
        self.max_steps = max_steps
        
        try:
            from datasets import load_dataset
            self.ds = load_dataset(dataset_name, split=split, streaming=True)
        except ImportError:
            raise ImportError("HuggingFace 'datasets' library not found. Please run: pip install datasets")

    def __iter__(self) -> Iterator[dict]:
        iterator = iter(self.ds)
        buffer = []
        
        for i, item in enumerate(iterator):
            if self.max_steps and i > self.max_steps:
                break
                
            text = item.get('text', '')
            tokens = self.tokenizer.encode(text)
            buffer.extend(tokens)
            
            # Yield chunks
            while len(buffer) >= self.block_size + 1:
                chunk = buffer[:self.block_size + 1]
                buffer = buffer[self.block_size:] # Sliding window? Or non-overlapping? 
                # Ideally non-overlapping for huge data, or stride.
                # Let's do stride = block_size (non-overlapping)
                
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y

class DataFactory:
    @staticmethod
    def get_loader(config, tokenizer, split='train'):
        name = config.dataset_name.lower()
        block_size = config.model.block_size
        batch_size = config.training.batch_size
        
        if name == 'tinystories' or name == 'tinyshakespeare':
            # Local file fallback
            file_path = f"data/{split}.txt"  # Assumes prepare_dataset structure
            if not os.path.exists(file_path):
                # Fallback to root input.txt for demo
                file_path = "input.txt"
            
            dataset = TextFileDataset(file_path, block_size, tokenizer)
            return DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'))
            
        elif name == 'fineweb-edu':
            # Use HuggingFace Streaming
            # "HuggingFaceFW/fineweb-edu"
            hf_name = "HuggingFaceFW/fineweb-edu"
            dataset = HFStreamDataset(hf_name, "train", block_size, tokenizer)
            return DataLoader(dataset, batch_size=batch_size) 
            
        else:
            raise ValueError(f"Unknown dataset: {name}")
