import torch
import argparse
import sys
import time
from config import GlobalConfig
from model_registry import ModelRegistry
from model_utils import SimpleTokenizer

def load_checkpoint(checkpoint_path, device='cpu'):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Recover config
    config = checkpoint.get('config')
    if config is None:
        print("Error: Config not found in checkpoint. Using default baseline config.")
        config = GlobalConfig()
    
    # Update device
    config.model.device = device
    
    # Build model
    model = ModelRegistry.build_model(config.model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config

def main():
    parser = argparse.ArgumentParser(description="DeepSeek Modular Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt or last_model.pt")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Starting text")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--device", type=str, default=None, help="Force device (cuda/cpu)")
    parser.add_argument("--vocab_path", type=str, default=None, help="Path to vocab.txt for SimpleTokenizer")
    args = parser.parse_args()

    # 1. Device Selection
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Handle RTX 50-series "no kernel image" issues
    try:
        model, config = load_checkpoint(args.checkpoint, device)
    except RuntimeError as e:
        if "no kernel image" in str(e) and device == 'cuda':
            print("WARNING: CUDA kernel missing (RTX 50-series). Falling back to CPU...")
            model, config = load_checkpoint(args.checkpoint, 'cpu')
            device = 'cpu'
        else:
            raise e

    # 2. Tokenizer Selection
    # If the training used fineweb-edu, it likely used gpt2 tokenizer
    # If it used local files, search for vocab.txt or assume SimpleTokenizer
    tokenizer = None
    if config.dataset_name == 'fineweb-edu':
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            print("Using HuggingFace GPT-2 Tokenizer.")
        except ImportError:
            print("Transformers not found. Cannot decode GPT-2 tokens.")
            sys.exit(1)
    else:
        # Try to load from vocab_path or default spots
        v_path = args.vocab_path or "vocab.txt"
        try:
            tokenizer = SimpleTokenizer.load_vocab(v_path)
            print(f"Loaded SimpleTokenizer from {v_path}")
        except FileNotFoundError:
            print(f"Warning: {v_path} not found. Using default charset (vulnerable to index errors).")
            tokenizer = SimpleTokenizer(text="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n.,!?'\"-")

    # 3. Generation
    print(f"\nGenerating {args.max_new_tokens} tokens on {device}...")
    print(f"Prompt: {args.prompt}")
    print("-" * 30)
    
    # Encode prompt
    if hasattr(tokenizer, 'encode'):
        idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    else:
        # HF tokenizer returns a dict or tensor
        idx = tokenizer.encode(args.prompt, return_tensors="pt").to(device)

    # Generation Loop (Streaming styles)
    start_time = time.time()
    generated_idx = idx
    
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            # Crop to block size
            idx_cond = generated_idx[:, -config.model.block_size:]
            
            # Forward
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / args.temperature
            
            # Top-K
            if args.top_k is not None:
                v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated_idx = torch.cat((generated_idx, idx_next), dim=1)
            
            # Decode and print (Streaming)
            step_token = idx_next[0].tolist()
            print(tokenizer.decode(step_token), end="", flush=True)

    end_time = time.time()
    print("\n" + "-" * 30)
    print(f"Generation finished in {end_time - start_time:.2f}s")

if __name__ == "__main__":
    main()
