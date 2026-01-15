import torch
import argparse
from config import GlobalConfig
from model_registry import ModelRegistry
from data_factory import DataFactory
from trainer import Trainer
from grpo_trainer import GRPOTrainer
from model_utils import SimpleTokenizer # Or standard tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='baseline', 
                       choices=['baseline', 'deepseek_v3', 'deepseek_r1'])
    parser.add_argument('--dataset', type=str, default='tinystories')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # 1. Config
    config = GlobalConfig()
    
    # Priority: GPU > CPU
    # If user asks for 'cuda' but no cuda, warn and switch.
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Switching to CPU.")
        real_device = 'cpu'
    else:
        real_device = args.device if torch.cuda.is_available() else 'cpu'

    config.model.device = real_device
    print(f"DEVICE SELECTED: {config.model.device.upper()}")
    if config.model.device == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    config.dataset_name = args.dataset
    
    # Adjust config based on model type
    if args.model_type == 'deepseek_v3':
        config.model.attention_type = 'mla'
        config.model.moe_type = 'deepseek'
        config.model.use_moe = True
    elif args.model_type == 'deepseek_r1':
        config.model.attention_type = 'mla'
        config.model.moe_type = 'deepseek'
        config.model.use_moe = True
        config.model.is_reasoning_model = True

    print(f"Training {args.model_type} on {args.dataset}...")
    
    # 2. Tokenizer (Assuming char-level for vanilla, or load real one)
    # Ideally should match dataset.
    if args.dataset == 'fineweb-edu':
        # Need a real tokenizer mostly, but for demo we can try loading one or fallback
        # For now, let's assume SimpleTokenizer can load a vocab file or built on fly (slow for huge data)
        # Use GPT-2 tokenizer usually for real stuff
        try:
             from transformers import AutoTokenizer
             tokenizer = AutoTokenizer.from_pretrained("gpt2")
             # patch encode/decode to be friendly
             config.model.vocab_size = tokenizer.vocab_size
        except:
             print("Transformers not found, using simple char tokenizer on tinystories default")
             tokenizer = SimpleTokenizer(text="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    else:
        # Load from file to build vocab
        tokenizer = SimpleTokenizer(text="abcdefghijklmnopqrstuvwxyz \n") # simplified

    # 3. Data
    train_loader = DataFactory.get_loader(config, tokenizer, split='train')
    
    # 4. Model
    model = ModelRegistry.build_model(config.model)
    model.to(config.model.device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 5. Trainer
    try:
        if config.model.is_reasoning_model:
            print("Using GRPOTrainer for Reasoning...")
            trainer = GRPOTrainer(model, None, config, tokenizer)
            # Train loop needs to be adapted for GRPO
            # Minimal loop:
            for i, (x, y) in enumerate(train_loader):
                loss = trainer.train_step(x.to(config.model.device))
                if i % 10 == 0:
                    print(f"Step {i}, GRPO Loss: {loss:.4f}")
                if i > 100: break # Demo limit
        else:
            print("Using Standard Trainer...")
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
            trainer = Trainer(model, config, optimizer, train_loader)
            trainer.train()
    except RuntimeError as e:
        if "no kernel image" in str(e):
            print("\nCRITICAL CUDA ERROR: Your GPU architecture (RTX 50-series/Blackwell) is too new for this PyTorch version.")
            print("Many CUDA kernels (Embedding, LayerNorm) are missing.")
            print(">>> AUTOMATICALLY SWITCHING TO CPU PREEMPTIVELY... <<<")
            
            # Switch to CPU
            config.model.device = 'cpu'
            model = model.to('cpu') # or rebuild? moving is fine
            
            print("Re-initializing Trainer on CPU...")
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
            
            # Re-create trainer
            if config.model.is_reasoning_model:
                 trainer = GRPOTrainer(model, None, config, tokenizer)
                 # We need to restart the loop logic for GRPO if we want to support it here
                 # For now, just warn GRPO might need manual restart or simple loop:
                 print("Restarting GRPO loop on CPU...")
                 for i, (x, y) in enumerate(train_loader):
                    loss = trainer.train_step(x.to('cpu'))
                    if i % 10 == 0: print(f"Step {i}, GRPO Loss: {loss:.4f}")
                    if i > 100: break
            else:
                 trainer = Trainer(model, config, optimizer, train_loader)
                 trainer.train()
        else:
            raise e

if __name__ == "__main__":
    main()
