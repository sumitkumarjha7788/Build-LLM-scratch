import torch
import torch.nn.functional as F
from model import GPTLanguageModel

def speculative_generate(
    target_model: GPTLanguageModel,
    draft_model: GPTLanguageModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    gamma: int = 4,
    temperature: float = 1.0,
    top_k: int = 0
):
    """
    Speculative Decoding Generator.
    
    Args:
        target_model: The large, accurate model used for verification.
        draft_model: The small, fast model used for drafting.
        idx: Initial context indices (B, T).
        max_new_tokens: Total tokens to generate.
        gamma: Number of speculative tokens to look ahead.
    
    Returns:
        Generated indices (B, T + max_new_tokens).
    """
    assert idx.shape[0] == 1, "Batch size must be 1 for current speculative implementation"
    
    T_orig = idx.shape[1]
    
    # We loop until we have generated enough tokens
    while idx.shape[1] < T_orig + max_new_tokens:
        
        # 1. Draft Step
        # Generate gamma tokens using the draft model
        # We need a custom generate loop that returns both tokens and their probs?
        # For simplicity, we just use the draft model's standard forward pass iteratively here.
        
        draft_tokens = torch.zeros((1, gamma), dtype=torch.long, device=idx.device)
        curr_idx = idx.clone()
        
        # We assume greedy decoding for simplicity in this template, 
        # or simple sampling if updated. Here implementing simple greedy/sample.
        
        for k in range(gamma):
            # Forward draft
            logits, _ = draft_model(curr_idx)
            logits = logits[:, -1, :]
            
            # Sample (or Greedy)
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            draft_tokens[:, k] = next_token
            curr_idx = torch.cat((curr_idx, next_token), dim=1)
            
        # 2. Target Verification Step
        # Run target model on [idx + draft_tokens]
        # We want to check the probabilities of the draft tokens given by the target model.
        
        # Input to target: Original context + Draft tokens
        # We need predictions for the *last* token of original context ... up to the last draft token.
        
        target_input = torch.cat((idx, draft_tokens), dim=1)
        target_logits, _ = target_model(target_input)
        
        # We are interested in logits at positions: [T-1, T, ..., T+gamma-1]
        # These correspond to predictions for: [draft_0, draft_1, ..., draft_gamma-1, valid_next]
        
        # Positions to verify:
        # We predicted draft_tokens[0] based on idx.
        # Target calculates prob of draft_tokens[0] given idx.
        
        # Let's verify acceptance
        n_accepted = 0
        current_pos = idx.shape[1]
        
        for k in range(gamma):
            # Token proposed by draft: draft_tokens[0, k]
            draft_token = draft_tokens[0, k]
            
            # Probability assigned by Target model to this token
            # Target output at [current_pos - 1 + k] predicts position [current_pos + k]
            # Wait, target_logits is (1, Len, Vocab).
            # Logits for predicting the token at `current_pos + k` come from `target_input` index `current_pos - 1 + k`
            
            logits_k = target_logits[:, current_pos - 1 + k, :]
            
            # Simplified Rejection Sampling (Greedy agreement for now to verify logic matches)
            # True speculative decoding uses : r = P_target(x) / P_draft(x)
            # Here we just check if Target *would have* chosen the same token (Greedy/Argmax) 
            # OR if we are doing sampling, we check the rejection criteria.
            
            # Implementing simple "Exact Match" verification for robustness in this phase
            target_token = torch.argmax(logits_k, dim=-1)
            
            if target_token == draft_token:
                n_accepted += 1
            else:
                # Rejected!
                # We stop accepting here.
                # Currently we take the Target's prediction as the correction.
                # correction_token = target_token
                break
        
        # 3. Update indices
        # We append the accepted tokens
        accepted_tokens = draft_tokens[:, :n_accepted]
        idx = torch.cat((idx, accepted_tokens), dim=1)
        
        # Add the correction token (prediction from Target at the failure point)
        # If we accepted all gamma, we still get one extra token from Target (the one after the draft sequence)
        
        # The logits for the "next" token after the last accepted one are available in target_logits
        # Index: current_pos - 1 + n_accepted
        if idx.shape[1] < T_orig + max_new_tokens:
             final_logits = target_logits[:, current_pos - 1 + n_accepted, :]
             if temperature > 0:
                 probs = F.softmax(final_logits / temperature, dim=-1)
                 correction_token = torch.multinomial(probs, num_samples=1)
             else:
                 correction_token = torch.argmax(final_logits, dim=-1, keepdim=True)
             
             idx = torch.cat((idx, correction_token), dim=1)
             
    return idx[:, :T_orig + max_new_tokens]
