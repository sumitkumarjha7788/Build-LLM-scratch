import torch
import torch.nn as nn
from torch.nn import functional as F
import os

from model_utils import ModelConfig
from model import GPTLanguageModel

# -----------------------------------------------------------------------------
# 1. Setup & Hyperparameters (Must match train.py exactly)
# -----------------------------------------------------------------------------
vocab_size = 65
block_size = 256
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
n_kv_head = 2 # Match GQA config
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

# (Vocabulary building section remains same)
file_path = "input.txt"
if not os.path.exists(file_path):
    print("Error: input.txt not found. Cannot build vocabulary.")
    exit(1)

with open(file_path, 'r', encoding='utf-8') as f:
    data = f.read()

chars = sorted(list(set(data)))
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(f"Vocabulary size: {len(chars)}")

# -----------------------------------------------------------------------------
# 3. Model Architecture
# -----------------------------------------------------------------------------
# Config must match training
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
    model = GPTLanguageModel(config)
    
    # Load state dict logic would go here. 
    # Warning: Current saved model is old architecture! It won't match.
    # We should re-train or warn the user.
    print("WARNING: Attempting to load model. If weights are from old architecture, this will fail.")
    try:
        # Strict=False to allow partial loading if experimenting, but usually requires full match
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict) 
    except Exception as e:
        print(f"FAILED to load model weights: {e}")
        print("You likely need to re-run train.py because the architecture has changed (RoPE/GQA added).")
        exit(1)
        
    model = model.to(device)
    model.eval() 
    print("Model loaded successfully.")
    
    print("\n" + "="*50)
    print("GPT-style Text Generator (GQA Enabled)")
    print("Enter a start phrase (or 'q' to quit)")
    print("="*50 + "\n")

    while True:
        prompt = input("Your prompt: ")
        if prompt.lower() in ['q', 'quit', 'exit']:
            break
            
        if not prompt.strip():
            prompt = "\n" 
            
        try:
            context_idxs = encode(prompt)
            context = torch.tensor(context_idxs, dtype=torch.long, device=device).unsqueeze(0) 
            
            print("\nGenerating...")
            generated_ids = model.generate(context, max_new_tokens=500)
            
            generated_text = decode(generated_ids[0].tolist())
            
            print("-" * 20 + " RESULT " + "-" * 20)
            print(generated_text)
            print("-" * 48 + "\n")
            
        except KeyError:
            print("Error: Invalid characters.")
        except Exception as e:
            print(f"Error: {e}")
