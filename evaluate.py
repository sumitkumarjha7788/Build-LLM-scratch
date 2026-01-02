import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import matplotlib.pyplot as plt
import os

from model_utils import ModelConfig
from model import GPTLanguageModel

# -----------------------------------------------------------------------------
# Setup (Must match train.py)
# -----------------------------------------------------------------------------
vocab_size = 65
block_size = 256
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
n_kv_head = 2 # GQA config
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ... (Analysis logic remains) ...

# -----------------------------------------------------------------------------
# 2. Qualitative Evaluation (Text Generation)
# -----------------------------------------------------------------------------
print("\n--- Model Outputs ---")

# Load Vocabulary
if not os.path.exists('input.txt'):
    print("input.txt not found, cannot build vocab.")
    exit()

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Model Config
config = ModelConfig(
    vocab_size=len(chars),
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout,
    n_kv_head=n_kv_head,
    use_flash_attention=True,
    device=device
)

# Load Model
model = GPTLanguageModel(config)
try:
    print("WARNING: Loading model weights. If these are from the old architecture, shapes will mismatch.")
    # Strict=False to handle potential missing keys if experimenting, but usually needs full match
    model.load_state_dict(torch.load('gpt_language_model.pt', map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Model load failed (likely architecture mismatch): {e}")
    # Don't exit, just skip generation so we can still see logs
    model = None

if model:
    # Generate Samples
    prompts = ["The", "To be", "KING:", "ROMEO:"]
    for p in prompts:
        print(f"\n[Prompt: '{p}']")
        try:
            context = torch.tensor(encode(p), dtype=torch.long, device=device).unsqueeze(0)
            out = model.generate(context, max_new_tokens=100)
            print(decode(out[0].tolist()))
        except Exception as e:
            print(f"Generation error: {e}")

