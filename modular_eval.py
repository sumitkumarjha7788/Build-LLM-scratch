import torch
import argparse
from config import GlobalConfig
from model_registry import ModelRegistry
from evaluation_suite import EvaluationSuite
from model_utils import SimpleTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to checkpoint")
    parser.add_argument('--model_type', type=str, default='baseline')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # 1. Config & Model
    config = GlobalConfig()
    config.model.model_type = args.model_type
    config.model.device = args.device if torch.cuda.is_available() else 'cpu'
    
    # Adjust config based on model type (same logic as training)
    if args.model_type == 'deepseek_v3':
        config.model.attention_type = 'mla'
        config.model.moe_type = 'deepseek'
        config.model.use_moe = True
    elif args.model_type == 'deepseek_r1':
        config.model.attention_type = 'mla'
        config.model.moe_type = 'deepseek'
        config.model.use_moe = True
    
    print(f"Loading {args.model_type} from {args.model_path}...")
    model = ModelRegistry.build_model(config.model)
    
    # Load weights
    # Allow for partial loading or strict loading depending on needs
    checkpoint = torch.load(args.model_path, map_location=config.model.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(config.model.device)
    model.eval()

    # 2. Tokenizer
    # TODO: Load from saved vocab or config
    tokenizer = SimpleTokenizer(text="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n")

    # 3. Eval Suite
    evaluator = EvaluationSuite(model, tokenizer)
    
    # 4. Run Core Evals
    print("Running Perplexity Check...")
    # Dummy text for now
    text_ids = torch.randint(0, config.model.vocab_size, (128,))
    ppl = evaluator.compute_perplexity(text_ids)
    print(f"Perplexity: {ppl:.4f}")
    
    print("Running Generic MMLU-style check...")
    qs = [
        {"question": "What is 1+1?", "choices": ["3", "2", "5"], "answer_idx": 1},
        {"question": "Capital of France?", "choices": ["Berlin", "London", "Paris"], "answer_idx": 2}
    ]
    acc = evaluator.multiple_choice_accuracy(qs)
    print(f"Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
