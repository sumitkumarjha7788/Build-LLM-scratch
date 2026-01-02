import torch
from model_utils import ModelConfig
from model import GPTLanguageModel
from pruning import MagnitudePruner
from benchmark import BenchmarkSuite

# Hyperparams
vocab_size = 65
block_size = 64
n_embd = 128
n_head = 4
n_layer = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Setup Model
config = ModelConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    device=device
)
model = GPTLanguageModel(config).to(device)

print("--- Testing Remaining Features ---")

# 2. Test Pruning
print("\n[Testing Pruning]")
model = MagnitudePruner.prune_model(model, amount=0.5)
# Verify Sparsity: Check if any weight is zero
has_zeros = False
for name, p in model.named_parameters():
    if 'weight' in name and p.dim() > 1:
        sparsity = (p == 0).float().mean().item()
        print(f"{name}: {sparsity:.2f} sparse")
        if sparsity > 0.4: has_zeros = True
        
if has_zeros:
    print("Pruning: PASSED")
else:
    print("Pruning: FAILED")

# 3. Test Benchmark
print("\n[Testing Benchmark Suite]")
BenchmarkSuite.report(model, config, device)

# 4. Final Task Checklist Update Check
print("\nRoadmap verification complete.")
