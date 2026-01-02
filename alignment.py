import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectPreferenceOptimization(nn.Module):
    """
    Direct Preference Optimization (DPO) Loss.
    Reference: https://arxiv.org/abs/2305.18290
    """
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(self, 
                policy_chosen_logps: torch.Tensor, 
                policy_rejected_logps: torch.Tensor, 
                ref_chosen_logps: torch.Tensor, 
                ref_rejected_logps: torch.Tensor):
        """
        Args:
            policy_chosen_logps: Log probs of chosen responses given by policy model. (B,)
            policy_rejected_logps: Log probs of rejected responses given by policy model. (B,)
            ref_chosen_logps: Log probs of chosen responses given by reference model. (B,)
            ref_rejected_logps: Log probs of rejected responses given by reference model. (B,)
        """
        # Calculate logits ratio
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        # DPO API: preference_loss = -logsigmoid(beta * (pi_logratios - ref_logratios))
        logits = pi_logratios - ref_logratios
        losses = -F.logsigmoid(self.beta * logits)
        
        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps).detach()

        return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()

class ChainOfThoughtTraining:
    """
    Utilities for CoT training, specifically masking.
    """
    @staticmethod
    def create_mask(input_ids: torch.Tensor, separation_token_id: int):
        """
        Creates a mask where the 'Question' part is masked (0) and 'Reasoning+Answer' is unmasked (1).
        Assumes data format: [User Question] [SEP] [Reasoning... Answer]
        """
        B, T = input_ids.shape
        mask = torch.ones((B, T), device=input_ids.device)
        
        # Find separation token for each sequence
        # This is a simplified version; real-world needs robust finding logic (e.g. first occurrence)
        sep_indices = (input_ids == separation_token_id).nonzero(as_tuple=False)
        
        # We might have multiple or no seps, handle carefully
        # Simple assumption: One sep per sequence
        
        for i in range(B):
            # Find first sep in this row
            row_sep = (sep_indices[:, 0] == i).nonzero()
            if row_sep.numel() > 0:
                sep_idx = sep_indices[row_sep[0], 1].item()
                # Mask everything up to and including separator (User prompt)
                mask[i, :sep_idx+1] = 0
                
        return mask
