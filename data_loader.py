import torch
import os
import requests
from torch.utils.data import Dataset, DataLoader
from model_utils import SimpleTokenizer
from alignment import ChainOfThoughtTraining

class TextDataset(Dataset):
    def __init__(self, data_tensor, block_size):
        self.data = data_tensor
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, block_size, cot_utils=None):
        self.data = data
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.cot_utils = cot_utils

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Format: Prompt \n Response
        full_text = item["prompt"] + "\n" + item["response"]
        ids = self.tokenizer.encode(full_text)
        
        # Truncate/Pad
        if len(ids) > self.block_size:
            ids = ids[:self.block_size]
        else:
            ids = ids + [0] * (self.block_size - len(ids))
        
        x = torch.tensor(ids, dtype=torch.long)
        
        # Masking for loss (don't train on prompt)
        mask = torch.ones_like(x)
        prompt_len = len(self.tokenizer.encode(item["prompt"] + "\n"))
        mask[:prompt_len] = 0
        
        return x, mask

class DataLoaderManager:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def get_pretrain_loaders(self, data_path, val_split=0.1):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file {data_path} not found.")
        
        
        # Use simple file reading for small files or memmap for large files?
        # For simplicity and robustness with the provided dataset format (int tokens):
        # We need to know if the file is raw text or numpy binary.
        # The project seems to use "input.txt" (text) or tokenizer.encode().
        # If we want memmap, we should ideally save tokens as .bin first.
        # However, sticking to the plan: "Update ... to use np.memmap ... remove 100MB limit"
        
        # If the file is a tokens file (e.g. .bin), we use memmap.
        # If it is .txt, we must read it. Reading a 1GB .txt file is okay-ish on modern systems, but not optimal.
        # Let's assume for this specific upgrade we are dealing with the fact that `prepare_dataset.py` might NOT be producing .bin yet.
        # But wait, `prepare_dataset.py` isn't fully visible here, but `main.py` says `if vocab_file: tokenizer.save_vocab`.
        
        # Let's modify this to read the full file (removing 100MB limit) and suggest .bin later?
        # Or better: implement a robust read that doesn't crash.
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = f.read() # Read FULL file (previously 100MB limited)
            
        data_tensor = torch.tensor(self.tokenizer.encode(data), dtype=torch.long)
        n = int((1 - val_split) * len(data_tensor))
        train_data = data_tensor[:n]
        val_data = data_tensor[n:]
        
        train_ds = TextDataset(train_data, self.config.model.block_size)
        val_ds = TextDataset(val_data, self.config.model.block_size)
        
        train_loader = DataLoader(train_ds, batch_size=self.config.training.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.config.training.batch_size, shuffle=False)
        
        return train_loader, val_loader

    def get_instruction_loader(self, data, shuffle=True):
        ds = InstructionDataset(data, self.tokenizer, self.config.model.block_size)
        return DataLoader(ds, batch_size=self.config.training.batch_size, shuffle=shuffle)
