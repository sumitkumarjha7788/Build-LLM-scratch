import torch
import torch.nn.functional as F
from model_utils import SimpleTokenizer
from model import GPTLanguageModel
from rag import AdaptiveRAG
from speculative import speculative_generate
from quantization import quantize_model
from pruning import MagnitudePruner

class InferenceEngine:
    def __init__(self, model_path, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.device = config.model.device
        
        # Load Model
        self.model = GPTLanguageModel(config.model)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Robust loading: check if it's a wrapped checkpoint or raw state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.rag = None
        if hasattr(config, 'rag_docs') and config.rag_docs:
            self.rag = AdaptiveRAG(config.rag_docs)

    def generate(self, prompt, max_new_tokens=100, use_rag=False, draft_model=None):
        if use_rag and self.rag:
            docs, confidence = self.rag.retrieve(prompt)
            prompt = self.rag.format_prompt(prompt, docs)
            print(f"RAG Augmented Prompt (Confidence: {confidence:.2f})")

        idx = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long, device=self.device).unsqueeze(0)
        
        if draft_model:
            print("Using Speculative Decoding...")
            generated_idx = speculative_generate(self.model, draft_model, idx, max_new_tokens)
        else:
            generated_idx = self.model.generate(idx, max_new_tokens)
            
        return self.tokenizer.decode(generated_idx[0].tolist())

    def apply_optimizations(self, prune_amount=0.0, use_quantization=False):
        if prune_amount > 0:
            MagnitudePruner.prune_model(self.model, amount=prune_amount)
        if use_quantization:
            self.model = quantize_model(self.model)
        self.model.to(self.device)
