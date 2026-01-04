import torch
import torch.nn.functional as F
from model import GPTLanguageModel
from model_utils import ModelConfig, SimpleTokenizer
import numpy as np

class EvaluationSuite:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def compute_perplexity(self, text_ids):
        """ Computes perplexity on a given sequence of token IDs. """
        self.model.eval()
        # Sliding window would be better for long text, but here we assume text_ids fits in block_size or we chunk it
        x = text_ids[:-1].unsqueeze(0).to(self.device)
        y = text_ids[1:].unsqueeze(0).to(self.device)
        
        logits, loss = self.model(x, y)
        return torch.exp(loss).item()

    @torch.no_grad()
    def multiple_choice_accuracy(self, questions):
        """
        Simplified MMLU style check.
        Format: [{"question": "What is 2+2?", "choices": ["3", "4", "5"], "answer_idx": 1}]
        """
        self.model.eval()
        correct = 0
        for q in questions:
            # We check which choice has the highest completion probability
            # Prob(Choice | Question)
            q_ids = self.tokenizer.encode(q["question"])
            
            choice_log_probs = []
            for choice in q["choices"]:
                c_ids = self.tokenizer.encode(choice)
                full_ids = torch.tensor(q_ids + c_ids, dtype=torch.long, device=self.device).unsqueeze(0)
                
                # Get logits for the choice tokens
                logits, _ = self.model(full_ids)
                # We need the logits at the positions of the choice tokens
                # Positions: [len(q_ids) - 1, ..., len(full_ids) - 2]
                # To predict choice tokens at: [len(q_ids), ..., len(full_ids) - 1]
                
                log_prob = 0
                for i in range(len(c_ids)):
                    actual_tok = c_ids[i]
                    target_pos = len(q_ids) - 1 + i
                    lp = F.log_softmax(logits[0, target_pos, :], dim=-1)
                    log_prob += lp[actual_tok].item()
                
                choice_log_probs.append(log_prob)
            
            pred_idx = np.argmax(choice_log_probs)
            if pred_idx == q["answer_idx"]:
                correct += 1
        
        return correct / len(questions)

if __name__ == "__main__":
    # Test with dummy model
    from config import ModelConfig
    config = ModelConfig(vocab_size=128, block_size=128, n_embd=128, n_head=4, n_layer=2)
    model = GPTLanguageModel(config)
    tokenizer = SimpleTokenizer(chars=[chr(i) for i in range(128)]) # full ASCII dummy
    
    suite = EvaluationSuite(model, tokenizer)
    
    # Dummy questions
    qs = [
        {"question": "A", "choices": ["B", "C"], "answer_idx": 0},
        {"question": "B", "choices": ["A", "C"], "answer_idx": 1}
    ]
    
    acc = suite.multiple_choice_accuracy(qs)
    print(f"Dummy Accuracy: {acc*100:.2f}%")
