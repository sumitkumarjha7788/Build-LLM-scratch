import torch
import time
import numpy as np

class BenchmarkSuite:
    """
    Provides metrics for LLM performance: Perplexity, Latency, Throughput.
    """
    @staticmethod
    def calculate_perplexity(model, text_ids, device):
        """ Returns the perplexity of the model on a given text sequence. """
        model.eval()
        with torch.no_grad():
            # x: all but last, y: all but first (causal)
            x = text_ids[:-1].unsqueeze(0).to(device)
            y = text_ids[1:].unsqueeze(0).to(device)
            
            logits, loss = model(x, y)
            perplexity = torch.exp(loss)
            return perplexity.item()

    @staticmethod
    def measure_latency(model, context_ids, tokens_to_gen=50, device='cuda'):
        """ Measures tokens per second. """
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            _ = model.generate(context_ids.to(device), tokens_to_gen)
            end_time = time.time()
            
            total_time = end_time - start_time
            tokens_per_sec = tokens_to_gen / total_time
            ms_per_token = (total_time / tokens_to_gen) * 1000
            
            return {
                "tokens_per_sec": tokens_per_sec,
                "ms_per_token": ms_per_token,
                "total_time": total_time
            }
            
    @staticmethod
    def report(model, config, device):
        print("\n" + "="*40)
        print("LLM BENCHMARK REPORT")
        print("="*40)
        
        # 1. Latency Test
        dummy_context = torch.zeros((1, 1), dtype=torch.long, device=device)
        metrics = BenchmarkSuite.measure_latency(model, dummy_context, device=device)
        print(f"Inference Latency: {metrics['ms_per_token']:.2f} ms/token")
        print(f"Throughput:       {metrics['tokens_per_sec']:.2f} tokens/sec")
        
        # 2. VRAM Usage
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated(0) / 1024**3
            print(f"VRAM Usage:        {vram:.2f} GB")
            
        print("="*40)
