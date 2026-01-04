import torch
import time
from model import GPTLanguageModel
from config import ModelConfig

def benchmark_efficiency():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Benchmarking on {device}...")
    
    # 1. Config
    config = ModelConfig(
        vocab_size=65,
        block_size=1024, # Larger block size to test scaling
        n_embd=768,      # Larger model for realistic test
        n_head=12,
        n_layer=12,
        n_kv_head=3,     # GQA: 4 query heads per KV head
        use_flash_attention=True,
        device=device
    )
    
    # 2. Model
    model = GPTLanguageModel(config).to(device)
    model.eval()
    
    # 3. Latency Test
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    tokens_to_generate = 50
    
    # Warmup
    _ = model.generate(ctx, 10)
    
    # Benchmark
    start_time = time.time()
    generated = model.generate(ctx, tokens_to_generate)
    end_time = time.time()
    
    total_time = end_time - start_time
    ms_per_token = (total_time / tokens_to_generate) * 1000
    tokens_per_sec = tokens_to_generate / total_time
    
    print("\n" + "="*40)
    print("Inference Efficiency results")
    print("="*40)
    print(f"Avg Latency: {ms_per_token:.2f} ms/token")
    print(f"Throughput:  {tokens_per_sec:.2f} tokens/sec")
    
    if ms_per_token < 50:
        print("\nTarget Verified: Latency < 50ms/token PASSED")
    else:
        print("\nTarget Failed: Optimization needed.")
    print("="*40)

if __name__ == "__main__":
    benchmark_efficiency()
