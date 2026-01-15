import torch
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Callable, List, Optional
import copy

class GRPOTrainer:
    """
    Group Relative Policy Optimization (GRPO) Trainer for DeepSeek-R1 capabilities.
    
    Algorithm:
    1. Sample G outputs for each prompt.
    2. Score them using an Oracle or Rule-based Reward.
    3. Compute Advantages: A_target = (Reward - Mean(Group)) / Std(Group).
    4. Update Policy: Maximize E[ min(ratio * A, clip * A) - beta * KL ]
    """
    def __init__(self, 
                 model: torch.nn.Module, 
                 ref_model: Optional[torch.nn.Module],
                 config, 
                 tokenizer):
        self.model = model
        self.ref_model = ref_model if ref_model else copy.deepcopy(model)
        # Freeze ref model
        for p in self.ref_model.parameters():
            p.requires_grad = False
            
        self.config = config
        self.tokenizer = tokenizer
        self.optimizer = AdamW(self.model.parameters(), lr=config.training.learning_rate) # Use standard LR from config
        self.beta = getattr(config.training, 'rlvr_beta', 0.04) # reuse or add new param
        self.group_size = getattr(config.training, 'group_size', 8)
        self.clip_eps = 0.2
        
    def compute_rewards(self, prompts: List[str], completions: List[str]) -> torch.Tensor:
        """
        Placeholder for specific reward logic (e.g. math correctness).
        """
        rewards = []
        for p, c in zip(prompts, completions):
            # Demo reward: Length of response (longer = better thinking? roughly mock)
            # In reality, check if answer is correct.
            reward = 0.0
            if "<answer>" in c:
                reward += 1.0
            rewards.append(reward)
        return torch.tensor(rewards, device=self.config.model.device)

    def train_step(self, prompts: torch.Tensor):
        """
        prompts: (B, T)
        """
        self.model.train()
        B, T = prompts.shape
        G = self.group_size
        
        # 1. Generate G outputs per prompt
        # Repeated prompts: (B*G, T)
        prompts_repeated = prompts.repeat_interleave(G, dim=0)
        
        # Generate is usually not differentiable. We generate text, then do a forward pass for probs.
        with torch.no_grad():
            outputs = self.model.generate(prompts_repeated, max_new_tokens=200) # (B*G, T+new)
            
        # Extract completions
        completion_ids = outputs[:, T:]
        
        # Decode for reward computation (if text based)
        # prompt_strs = [self.tokenizer.decode(p) for p in prompts_repeated]
        # completion_strs = [self.tokenizer.decode(c) for c in completion_ids]
        # rewards = self.compute_rewards(prompt_strs, completion_strs) # (B*G)
        
        # Mock rewards for tensor-only run
        rewards = torch.randn(B * G, device=prompts.device)
        
        # 2. Compute Group Advantages
        # Reshape to (B, G) to compute stats per group
        rewards_grouped = rewards.view(B, G)
        mean_rewards = rewards_grouped.mean(dim=1, keepdim=True)
        std_rewards = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards_grouped - mean_rewards) / std_rewards # (B, G)
        advantages = advantages.view(-1) # (B*G)
        
        # 3. Policy Update
        # Forward pass on generated outputs
        logits, _ = self.model(outputs) # (B*G, Len, Vocab)
        # We only care about logits for the completion part
        # logits: shift left? usually model returns prediction for next token.
        # logits[:, :-1] predicts outputs[:, 1:]
        
        # Align logits and labels
        # We want probability of completion_ids
        # logits corresponding to T-1 ... End-1 predict T ... End
        gen_logits = logits[:, T-1 : -1, :]
        gen_ids = outputs[:, T:]
        
        # Log Probs
        log_probs = F.log_softmax(gen_logits, dim=-1)
        token_log_probs = torch.gather(log_probs, 2, gen_ids.unsqueeze(-1)).squeeze(-1) # (B*G, Len_gen)
        
        # Ref Model Log Probs (for KL)
        with torch.no_grad():
            ref_logits, _ = self.ref_model(outputs)
            ref_gen_logits = ref_logits[:, T-1 : -1, :]
            ref_log_probs = F.log_softmax(ref_gen_logits, dim=-1)
            ref_token_log_probs = torch.gather(ref_log_probs, 2, gen_ids.unsqueeze(-1)).squeeze(-1)
            
        # KL Divergence (approx as difference in log probs: log(pi/ref) = log_pi - log_ref)
        # D_kl = sum(exp(pi) * (log_pi - log_ref)) ... straightforward estimator is just ratio
        # PPO usually uses ratio. GRPO paper uses KL penalty term directly.
        # deepseek paper: - beta * D_kl.
        # here we implement the per-token KL approx for gradient: pi/ref - 1 ? No, just log_pi - log_ref
        
        kl_div = torch.exp(token_log_probs - ref_token_log_probs) - (token_log_probs - ref_token_log_probs) - 1
        # Approx: (pi/ref) - log(pi/ref) - 1 ... Schulman approx
        # Simpler: log_pi - log_ref
        per_token_kl = token_log_probs - ref_token_log_probs
        
        # Importance Sampling Ratio (if we did multiple epochs on same samples, but GRPO is usually online? 
        # Paper implies simple advantage optimization).
        # We'll treat the generated samples as "current policy" implies ratio=1 initially.
        # If we do multiple update steps, we need 'old_log_probs'.
        
        # Let's assume strict online 1-step for simplicity or recalculate 'old' if loop.
        ratio = torch.exp(token_log_probs - token_log_probs.detach()) # = 1.0 first pass
        
        # Loss per token? 
        # Advantage is per SEQUENCE (B*G). We broadcast it to tokens? 
        # Usually RL in LLMs applies advantage to all tokens in response.
        
        advantages_expanded = advantages.unsqueeze(1).repeat(1, token_log_probs.shape[1])
        
        # GRPO Objective
        surr1 = ratio * advantages_expanded
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_expanded
        policy_loss = -torch.min(surr1, surr2).mean()
        
        kl_loss = self.beta * per_token_kl.mean()
        
        loss = policy_loss + kl_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
