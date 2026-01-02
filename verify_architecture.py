import torch
from model_utils import ModelConfig
from model import GPTLanguageModel

# Hyperparams
vocab_size = 65
block_size = 256
n_embd = 384
n_head = 6
n_layer = 4
n_kv_head = 2 # GQA: 3x reduction in KV cache
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Testing on device: {device}")

# 1. Init Config
config = ModelConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    n_kv_head=n_kv_head,
    use_flash_attention=True,
    device=device
)

print(f"Config initialized.")
print(f"GQA Ratios: n_head={n_head}, n_kv_head={n_kv_head}")

# 2. Init Model
try:
    model = GPTLanguageModel(config)
    model.to(device)
    print("Model initialized successfully.")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
except Exception as e:
    print(f"Model initialization FAILED: {e}")
    exit(1)

# 3. Forward Pass Test
try:
    B, T = 8, 128
    x = torch.randint(0, vocab_size, (B, T)).to(device)
    targets = torch.randint(0, vocab_size, (B, T)).to(device)
    
    print(f"Running forward pass with batch={B}, seq_len={T}...")
    logits, loss = model(x, targets)
    
    print(f"Forward pass successful.")
    print(f"Logits shape: {logits.shape} (Expected: {B, T, vocab_size})")
    print(f"Loss: {loss.item():.4f}")
    
except Exception as e:
    print(f"Forward pass FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\nPhase 1 Verification: PASSED")
