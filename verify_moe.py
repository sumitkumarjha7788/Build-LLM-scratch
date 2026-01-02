import torch
from model_utils import ModelConfig
from model import GPTLanguageModel

# Hyperparams
vocab_size = 65
block_size = 256
n_embd = 384
n_head = 6
n_layer = 4
n_kv_head = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Testing MoE on device: {device}")

# 1. Init Config with MoE ENABLED
config = ModelConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    n_kv_head=n_kv_head,
    use_flash_attention=True,
    use_moe=True, # Toggle ON
    num_experts=8,
    num_experts_per_tok=2,
    device=device
)

print(f"Config initialized.")
print(f"MoE Settings: Experts={config.num_experts}, Active={config.num_experts_per_tok}")

# 2. Init Model
try:
    model = GPTLanguageModel(config)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("Model initialized successfully.")
    print(f"Total Parameters (Dense + Sparse): {total_params/1e6:.2f} M")
    
    # Calculate active parameters (approx)
    # Dense parts: Embeddings, Attention, LayerNorms, Head
    # MoE parts: Router + Active Experts (2/8)
    # This is a rough check to see if params increased vs base model (~6M)
    if total_params < 10000000: # Base was ~6M. MoE should replace FFN (4*dim^2) with 8 * (4*dim^2)
        print("WARNING: Parameter count seems low for MoE. Did it verify?")
    else:
        print("Parameter count looks correct (significantly higher than 6M base).")

except Exception as e:
    print(f"Model initialization FAILED: {e}")
    exit(1)

# 3. Forward Pass Test
try:
    B, T = 4, 64
    x = torch.randint(0, vocab_size, (B, T)).to(device)
    targets = torch.randint(0, vocab_size, (B, T)).to(device)
    
    print(f"Running MoE forward pass with batch={B}, seq_len={T}...")
    logits, loss = model(x, targets)
    
    print(f"Forward pass successful.")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
except Exception as e:
    print(f"Forward pass FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\nPhase 3 (MoE) Verification: PASSED")
