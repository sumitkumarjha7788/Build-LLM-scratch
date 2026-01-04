import argparse
import torch
import os
from config import GlobalConfig, ModelConfig, TrainingConfig
from model_utils import SimpleTokenizer
from model import GPTLanguageModel
from data_loader import DataLoaderManager
from trainer import Trainer
from inference import InferenceEngine

def main():
    parser = argparse.ArgumentParser(description="Unified LLM Training & Optimization System")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "generate", "prepare"], help="Action to perform")
    parser.add_argument("--dataset", type=str, default="tinystories", help="Small (tinyshakespeare) or large (tinystories)")
    parser.add_argument("--iters", type=int, default=5000, help="Number of training iterations")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training or for generation")
    parser.add_argument("--prompt", type=str, default="\n", help="Prompt for generation")
    parser.add_argument("--use_moe", action="store_true", help="Enable Mixture of Experts")
    parser.add_argument("--num_experts", type=int, default=4, help="Number of experts for MoE")
    parser.add_argument("--prune", type=float, default=0.0, help="Pruning amount (0.0 to 1.0)")
    parser.add_argument("--quantize", action="store_true", help="Enable 4-bit quantization")
    
    args = parser.parse_args()
    
    # 1. Load Config
    config = GlobalConfig()
    config.training.max_iters = args.iters
    config.dataset_name = args.dataset
    config.model.use_moe = args.use_moe
    config.model.num_experts = args.num_experts
    config.model.num_experts_per_tok = 2 # Keeping this as sensible default
    # 2. Tokenizer & Data
    data_file = "input.txt" if args.dataset == "tinyshakespeare" else "tinystories.txt"
    vocab_file = "vocab.txt" if args.dataset == "tinystories" else None
    
    if args.dataset == "tinystories" and not os.path.exists(data_file):
        print("Please run prepare_dataset.py first or set mode to 'prepare'")
        if args.mode != "prepare": return

    if vocab_file and os.path.exists(vocab_file):
        tokenizer = SimpleTokenizer.load_vocab(vocab_file)
    else:
        # Fallback or build from sample
        with open(data_file, 'r', encoding='utf-8') as f:
            sample = f.read(1000000)
        tokenizer = SimpleTokenizer(text=sample)
        if vocab_file: tokenizer.save_vocab(vocab_file)

    config.model.vocab_size = tokenizer.vocab_size
    
    # 3. Mode Selection
    if args.mode == "prepare":
        from prepare_dataset import prepare_tinystories
        prepare_tinystories()
        
    elif args.mode == "train":
        # Initialize Model
        model = GPTLanguageModel(config.model)
        model.to(config.model.device)
        
        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
        
        # Data
        dl_manager = DataLoaderManager(config, tokenizer)
        train_loader, val_loader = dl_manager.get_pretrain_loaders(data_file)
        
        # Trainer
        trainer = Trainer(model, config, optimizer, train_loader, val_loader)
        
        if args.resume:
            trainer.load_checkpoint(args.resume)
            
        trainer.train()
        
    elif args.mode == "generate":
        if not args.resume:
            print("Error: --resume [checkpoint_path] is required for generate mode.")
            return
            
        engine = InferenceEngine(args.resume, config, tokenizer)
        
        # Apply optimizations if requested
        if args.prune > 0 or args.quantize:
            engine.apply_optimizations(prune_amount=args.prune, use_quantization=args.quantize)
        
        print("\nGenerating...")
        print(engine.generate(args.prompt))

if __name__ == "__main__":
    main()
