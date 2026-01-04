import torch
import torch.nn as nn
from torch.nn import functional as F
import requests
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Numpy version: {np.__version__}")

# Robust GPU Check
if torch.cuda.is_available():
    device = 'cuda'
    print(f"\nGPU Available: Yes (CUDA)")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
else:
    device = 'cpu'
    print("\nGPU Available: No, using CPU. Warning: Training will be slow.")

# Hyperparameters
batch_size = 64 # How many independent sequences will we process in parallel?
block_size = 256 # What is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

from model_utils import ModelConfig, SimpleTokenizer
from model import GPTLanguageModel

# Data Preparation
dataset_name = "tinystories" # Set to "tinyshakespeare" or "tinystories"
if dataset_name == "tinyshakespeare":
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    file_path = "input.txt"
    if not os.path.exists(file_path):
        print("Downloading tinyshakespeare...")
        data = requests.get(url).text
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(data)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    tokenizer = SimpleTokenizer(text=data)
elif dataset_name == "tinystories":
    file_path = "tinystories.txt"
    vocab_path = "vocab.txt"
    if not os.path.exists(file_path):
        print("TinyStories not found. Please run prepare_dataset.py first.")
        # Fallback to tinyshakespeare for demo if needed, but here we expect the user to run it
        exit(1)
    
    tokenizer = SimpleTokenizer.load_vocab(vocab_path)
    # For large datasets, we might not want to load everything into memory
    # But for this script, we'll try to read a reasonable chunk or handle it if it fits
    print("Loading TinyStories...")
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read first 100MB for training if it's too large, or whole file if possible
        data = f.read(100 * 1024 * 1024) 

vocab_size = tokenizer.vocab_size
encode = tokenizer.encode
decode = tokenizer.decode

# Data Splits
data_tensor = torch.tensor(encode(data), dtype=torch.long)
n = int(0.9 * len(data_tensor))
train_data = data_tensor[:n]
val_data = data_tensor[n:]

# Data Loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

from model_utils import ModelConfig
from model import GPTLanguageModel

# Hyperparameters (Controlled via Config now, but keeping variables for script logic)
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# GQA specific
n_kv_head = 2 # Example: 6 query heads, 2 kv heads (3x sharing)

# Initialize Config and Model
config = ModelConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout,
    n_kv_head=n_kv_head, # Enable GQA
    use_flash_attention=True
)

if __name__ == '__main__':
    model = GPTLanguageModel(config)
    m = model.to(device)
    # print the number of parameters in the model
    print(f"Model parameters: {sum(p.numel() for p in m.parameters())/1e6:.2f} M")

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Prepare logging
    log_file = "training_log.csv"
    with open(log_file, "w") as f:
        f.write("step,train_loss,val_loss\n")

    start_time = time.time()

    print("Starting training...")
    for iter in range(max_iters):
        if iter % 10 == 0: print(f"Iter {iter}...") # Heartbeat

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            print(f"Evaluating loss at step {iter}...")
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Log to file
            with open(log_file, "a") as f:
                f.write(f"{iter},{losses['train']:.4f},{losses['val']:.4f}\n")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"Training finished in {(time.time() - start_time)/60:.2f} minutes.")
    
    # Save the model
    torch.save(model.state_dict(), 'gpt_language_model.pt')
    print("Model saved to 'gpt_language_model.pt'")

    # Validation Plot
    try:
        df = pd.read_csv(log_file)
        plt.figure(figsize=(10, 6))
        plt.plot(df['step'], df['train_loss'], label='Train Loss')
        plt.plot(df['step'], df['val_loss'], label='Val Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_loss.png')
        print("Training loss plot saved to training_loss.png")
    except Exception as e:
        print(f"Could not save plot: {e}")

    # generate from the model
    print("\nGenerating text:")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
