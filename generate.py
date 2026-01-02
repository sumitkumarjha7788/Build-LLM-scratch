import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# -----------------------------------------------------------------------------
# 1. Setup & Hyperparameters (Must match train.py exactly)
# -----------------------------------------------------------------------------
batch_size = 64
block_size = 256
max_iters = 5000 # Not used for inference
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

# -----------------------------------------------------------------------------
# 2. Vocabulary & Encoding (Must match training data)
# -----------------------------------------------------------------------------
file_path = "input.txt"
if not os.path.exists(file_path):
    print("Error: input.txt not found. Cannot build vocabulary.")
    print("Please run train.py first to download the dataset.")
    exit(1)

with open(file_path, 'r', encoding='utf-8') as f:
    data = f.read()

chars = sorted(list(set(data)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(f"Vocabulary size: {vocab_size}")

# -----------------------------------------------------------------------------
# 3. Model Architecture (Identical to train.py)
# -----------------------------------------------------------------------------
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

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
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head=n_head) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_f(x)
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx

# -----------------------------------------------------------------------------
# 4. Main Inference Loop
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    model_path = 'gpt_language_model.pt'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please run train.py to train and save the model first.")
        exit(1)

    print(f"Loading model from {model_path}...")
    model = GPTLanguageModel()
    
    # Load state dict
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit(1)
        
    model = model.to(device)
    model.eval() # Set to evaluation mode
    print("Model loaded successfully.")
    
    print("\n" + "="*50)
    print("GPT-style Text Generator")
    print("Enter a start phrase (or 'q' to quit)")
    print("="*50 + "\n")

    while True:
        prompt = input("Your prompt: ")
        if prompt.lower() in ['q', 'quit', 'exit']:
            break
            
        if not prompt.strip():
            prompt = "\n" # Use newline if empty
            
        # Encode
        try:
            context_idxs = encode(prompt)
            context = torch.tensor(context_idxs, dtype=torch.long, device=device).unsqueeze(0) # (1, T)
            
            print("\nGenerating...")
            # Generate
            generated_ids = model.generate(context, max_new_tokens=500)
            
            # Decode
            generated_text = decode(generated_ids[0].tolist())
            
            print("-" * 20 + " RESULT " + "-" * 20)
            print(generated_text)
            print("-" * 48 + "\n")
            
        except KeyError:
            print("Error: Your prompt contains characters not in the training vocabulary (Tiny Shakespeare).")
            print("Try using simple English letters and punctuation.\n")
        except Exception as e:
            print(f"Error during generation: {e}\n")
