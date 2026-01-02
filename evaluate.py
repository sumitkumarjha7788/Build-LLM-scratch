import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------------------------------
# Setup (Must match train.py)
# -----------------------------------------------------------------------------
batch_size = 64
block_size = 256
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------------------------------------------------------
# 1. Quantitative Evaluation (Loss Analysis)
# -----------------------------------------------------------------------------
print("--- Training Metrics ---")
try:
    df = pd.read_csv('training_log.csv')
    print(f"Total Steps: {len(df) * 500}") # Approximate based on save interval if row count is small
    print(f"Final Train Loss: {df.iloc[-1]['train_loss']:.4f}")
    print(f"Final Val Loss:   {df.iloc[-1]['val_loss']:.4f}")
    print(f"Minimum Val Loss: {df['val_loss'].min():.4f} at step {df.loc[df['val_loss'].idxmin()]['step']}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['step'], df['train_loss'], label='Train Loss')
    plt.plot(df['step'], df['val_loss'], label='Val Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('eval_loss_plot.png')
    print("Loss plot saved to 'eval_loss_plot.png'")
    
except Exception as e:
    print(f"Could not read training logs: {e}")

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
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Model Definition
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x); q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x); out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size); self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd); self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx); pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb; x = self.blocks(x); x = self.ln_f(x); logits = self.lm_head(x)
        return logits, None
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Load Model
model = GPTLanguageModel()
try:
    model.load_state_dict(torch.load('gpt_language_model.pt', map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Model load failed: {e}")
    exit()

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
