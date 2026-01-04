import torch
import torch.optim as optim
from model_utils import ModelConfig
from model import GPTLanguageModel
from pruning import MagnitudePruner
from quantization import quantize_model
from benchmark import BenchmarkSuite
from rag import AdaptiveRAG
from alignment import DirectPreferenceOptimization
from rlvr import RLVRTrainer

# Hyperparams
vocab_size = 65
block_size = 64
n_embd = 128
n_head = 4
n_layer = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_final_verification():
    print("="*60)
    print("      LLM ARCHITECTURE UPGRADE: GRAND FINAL VERIFICATION")
    print("="*60)
    results = {}

    # 1. Setup Model
    config = ModelConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        device=device,
        use_moe=True,
        num_experts=4,
        num_experts_per_tok=2
    )
    model = GPTLanguageModel(config).to(device)
    print(f"[*] Model initialized with MoE and RoPE. Device: {device}")

    # 2. Test MoE & Forward Pass
    try:
        x = torch.randint(0, vocab_size, (1, block_size)).to(device)
        logits, _ = model(x)
        results["Core Architecture & MoE"] = "PASSED"
    except Exception as e:
        results["Core Architecture & MoE"] = f"FAILED: {e}"

    # 3. Test Alignment (DPO Logics)
    try:
        dpo = DirectPreferenceOptimization()
        # Mock logits
        p_c, p_r = torch.randn(1), torch.randn(1)
        r_c, r_r = torch.randn(1), torch.randn(1)
        dpo(p_c, p_r, r_c, r_r)
        results["Alignment (DPO)"] = "PASSED"
    except Exception as e:
        results["Alignment (DPO)"] = f"FAILED: {e}"

    # 4. Test RAG
    try:
        rag = AdaptiveRAG()
        rag.add_document("The quick brown fox.")
        docs, _ = rag.retrieve("fox")
        if "fox" in docs[0]:
            results["Adaptive RAG"] = "PASSED"
        else:
            results["Adaptive RAG"] = "FAILED (Retrieval mismatch)"
    except Exception as e:
        results["Adaptive RAG"] = f"FAILED: {e}"

    # 5. Test Optimization (Pruning & Quantization)
    try:
        # Pruning
        model = MagnitudePruner.prune_model(model, amount=0.2)
        # Quantization
        model = quantize_model(model)
        results["Efficiency (Prune/Quant)"] = "PASSED"
    except Exception as e:
        results["Efficiency (Prune/Quant)"] = f"FAILED: {e}"

    # 6. Benchmark
    print("\n[Running Benchmark]")
    try:
        BenchmarkSuite.report(model, config, device)
        results["Performance Benchmark"] = "PASSED"
    except Exception as e:
        results["Performance Benchmark"] = f"FAILED: {e}"

    # --- FINAL REPORT ---
    print("\n" + "="*60)
    print(f"{'FEATURE':<30} | {'STATUS':<20}")
    print("-" * 60)
    for feature, status in results.items():
        print(f"{feature:<30} | {status:<20}")
    print("="*60)
    
    if all(s == "PASSED" for s in results.values()):
        print("\nALL PHASES VERIFIED SUCCESSFULLY. ROADMAP COMPLETE.")
    else:
        print("\nSOME PHASES FAILED VERIFICATION. CHECK LOGS.")

if __name__ == "__main__":
    run_final_verification()

