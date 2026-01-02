import torch
import time
from model_utils import ModelConfig
from model import GPTLanguageModel
from speculative import speculative_generate

# Setup
vocab_size = 65
block_size = 64
n_embd = 128
n_head = 4
n_layer = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Testing Scaling on device: {device}")

# -----------------------------------------------------------------------------
# Test 1: RoPE Scaling
# -----------------------------------------------------------------------------
print("\n--- Test 1: RoPE Scaling ---")
config_scaled = ModelConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    rope_scaling_factor=2.0, # Test scaling
    device=device
)

try:
    model_scaled = GPTLanguageModel(config_scaled).to(device)
    print("Model with Scaling Factor=2.0 initialized.")
    x = torch.randint(0, vocab_size, (1, block_size)).to(device)
    out, _ = model_scaled(x)
    print(f"Forward pass successful. Output shape: {out.shape}")
except Exception as e:
    print(f"RoPE Scaling FAILED: {e}")
    exit(1)

# -----------------------------------------------------------------------------
# Test 2: Speculative Decoding
# -----------------------------------------------------------------------------
print("\n--- Test 2: Speculative Decoding ---")
# Use same config for Target and Draft for verification (simulating perfect draft)
config_spec = ModelConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    device=device
)

target_model = GPTLanguageModel(config_spec).to(device)
draft_model = target_model # self-speculation for test

idx = torch.zeros((1, 1), dtype=torch.long, device=device) # Start with token 0

try:
    print("Running speculative generation...")
    start = time.time()
    # Generate 20 tokens, lookahead gamma=4
    output = speculative_generate(
        target_model, draft_model, idx, max_new_tokens=20, gamma=4, temperature=0
    )
    end = time.time()
    
    print(f"Speculative generation successful.")
    print(f"Generated sequence length: {output.shape[1]}")
    print(f"Time taken: {end - start:.4f}s")
    
except Exception as e:
    print(f"Speculative Decoding FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\nPhase 3 (Scaling) Verification: PASSED")
