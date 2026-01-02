import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import time
import os
from model_utils import ModelConfig
from model import GPTLanguageModel
from alignment import ChainOfThoughtTraining

# SFT Configuration
sft_batch_size = 8
sft_learning_rate = 1e-5
sft_epochs = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dummy SFT Dataset (Format: User -> Assistant)
# In real scenario, load from JSONL
dummy_data = [
    {"prompt": "What is 2+2?", "response": "2+2 is 4."},
    {"prompt": "Who wrote Hamlet?", "response": "Shakespeare wrote Hamlet."},
    {"prompt": "Define AI.", "response": "AI stands for Artificial Intelligence."},
    # ... repeated for testing
] * 10

# Simple Tokenizer (Consistent with project)
if not os.path.exists("input.txt"):
    print("Error: input.txt needed for vocab")
    exit(1)
with open("input.txt", 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi.get(c, 0) for c in s] # Handle unknown chars gracefully
decode = lambda l: ''.join([itos[i] for i in l])
vocab_size = len(chars)

class SFTDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        # Format: Prompt [SEP] Response
        # We assume a separator token, or just raw concat for Shakespeare style
        # For this demo, let's just concat with a newline
        full_text = item["prompt"] + "\n" + item["response"]
        
        # Encode
        ids = encode(full_text)
        
        # Truncate/Pad
        if len(ids) > self.block_size:
            ids = ids[:self.block_size]
        else:
            # Pad with 0? Or just leave it variable length and collate?
            # Model expects fixed block_size for positional embeddings if using absolute, 
            # but RoPE handles variable. However, batching needs padding.
            ids = ids + [0] * (self.block_size - len(ids))
            
        x = torch.tensor(ids, dtype=torch.long)
        
        # Masking: We want to mask the Prompt loss? 
        # Identify split point
        prompt_len = len(encode(item["prompt"] + "\n"))
        mask = torch.ones_like(x)
        mask[:prompt_len] = 0 # Don't train on prompt
        
        return x, mask

# Init Model
config = ModelConfig(
    vocab_size=vocab_size,
    block_size=256,
    n_embd=384,
    n_head=6,
    n_layer=6,
    n_kv_head=2,
    use_flash_attention=True,
    device=device
)

# Training Loop
def train_sft():
    print("--- Starting SFT Training ---")
    model = GPTLanguageModel(config)
    model.to(device)
    
    # Load Pre-trained weights if available
    if os.path.exists('gpt_language_model.pt'):
        print("Loading pre-trained weights...")
        try:
            model.load_state_dict(torch.load('gpt_language_model.pt', map_location=device))
        except:
            print("Warning: Weights mismatch, starting fresh (OK for testing)")
            
    optimizer = torch.optim.AdamW(model.parameters(), lr=sft_learning_rate)
    dataset = SFTDataset(dummy_data, config.block_size)
    dataloader = DataLoader(dataset, batch_size=sft_batch_size, shuffle=True)
    
    model.train()
    
    for epoch in range(sft_epochs):
        for batch_idx, (x, mask) in enumerate(dataloader):
            x = x.to(device)
            mask = mask.to(device)
            
            # Standard forward - Model calculates loss on all targets automatically
            # We need manual loss to apply mask
            logits, _ = model(x) 
            
            # Shift for autoregressive loss
            # logits: (B, T, V) -> predict next
            # targets: (B, T) -> x (input is same as target in causal LM)
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = x[..., 1:].contiguous()
            shift_mask = mask[..., 1:].contiguous()
            
            # Cross Entropy (Reduction=None to apply mask)
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
            
            # Apply mask
            B_curr = x.shape[0]
            loss = loss.view(B_curr, -1)
            loss = (loss * shift_mask).sum() / shift_mask.sum()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 2 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    print("SFT Training Complete.")
    torch.save(model.state_dict(), 'sft_model.pt')
    print("Saved 'sft_model.pt'")

if __name__ == '__main__':
    train_sft()
