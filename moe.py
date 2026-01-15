import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import ModelConfig

class Expert(nn.Module):
    """ An individual expert is a standard FeedForward network """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class DynamicRouterMoE(nn.Module):
    """
    Dynamic Router for Mixture of Experts.
    Routes each token to the top-k experts.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_embd = config.n_embd
        
        # The gating network
        self.router = nn.Linear(self.n_embd, self.num_experts, bias=False)
        
        # The experts
        self.experts = nn.ModuleList([Expert(config) for _ in range(self.num_experts)])

    def forward(self, x):
        # x shape: (Batch, Time, Channels)
        B, T, C = x.shape
        
        # Flatten batch and time for routing: (B*T, C)
        x_flat = x.view(-1, C)
        
        # Compute router logits: (B*T, num_experts)
        router_logits = self.router(x_flat)
        
        # Select top-k experts
        # routing_weights: (B*T, k)
        # selected_experts: (B*T, k)
        routing_weights, selected_experts = torch.topk(router_logits, self.num_experts_per_tok, dim=-1)
        
        # Normalize weights so they sum to 1
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float).to(x.dtype)
        
        # Initialize output tensor
        # We need to accumulate results from different experts.
        # Efficient "Scatter/Gather" implementation usually used in big frameworks.
        # Here we use a simpler iterative approach for clarity and flexibility logic.
        
        final_output = torch.zeros_like(x_flat)
        
        # We iterate over the k selected experts. 
        # This is a bit slow in python loop but acceptable for k=2 and experimental scale.
        # For production training, one would optimize with masked matmuls or specialized CUDA kernels.
        
        # Create a mask for each expert
        # One-hot-ish approach (but we only have k per token)
        
        # Let's iterate over ALL experts and mask operations.
        # This is safe and robust but computes "zero" for unselected tokens (wasted compute check?).
        # Actually, iterating over ALL experts is efficient if batch size is large enough that every expert gets data.
        # But if we want TRUE sparsity (skipping compute), we must only run selected.
        
        # Approach: Loop over k choices
        for k in range(self.num_experts_per_tok):
            # Get the k-th selected expert index for each token
            # selected_experts_k: (B*T)
            expert_indices = selected_experts[:, k]
            weights = routing_weights[:, k].unsqueeze(1) # (B*T, 1)
            
            # For each unique expert selected in this 'k' slot
            for i, expert in enumerate(self.experts):
                # Identify which tokens selected expert 'i' as their k-th choice
                # Using a mask
                token_mask = (expert_indices == i)
                
                if token_mask.any():
                    # Select the tokens
                    selected_tokens = x_flat[token_mask]
                    
                    # Run expert
                    expert_out = expert(selected_tokens)
                    
                    # Add weighted contribution to final output
                    # We need to map back to the original positions.
                    # We can use the mask index.
                    
                    # Accumulate: final_output[mask] += weight * expert_out
                    # Note: In-place addition with boolean indexing works in PyTorch
                    final_output[token_mask] += weights[token_mask] * expert_out


        return final_output.view(B, T, C)

class DeepSeekMoE(nn.Module):
    """
    DeepSeek-V3 MoE Architecture.
    Key features:
    1. Shared Experts: Constantly active experts for common knowledge.
    2. Fine-Grained Routed Experts: Select top-k from many small experts.
    3. Aux-Less Load Balancing: Bias-based routing update (Simplified version).
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_shared_experts = getattr(config, 'num_shared_experts', 1)
        self.n_embd = config.n_embd
        
        # 1. Shared Experts (Always active)
        self.shared_experts = nn.ModuleList([Expert(config) for _ in range(self.num_shared_experts)])
        
        # 2. Routed Experts (Selected via Top-K)
        self.routed_experts = nn.ModuleList([Expert(config) for _ in range(self.num_experts)])
        
        # 3. Router
        self.router = nn.Linear(self.n_embd, self.num_experts, bias=False)
        
        # 4. Bias for Load Balancing (Aux-free)
        # We learn a bias term that adjusts the probability of selecting an expert.
        # In full DeepSeek-V3, this is updated via a specific rule, but here we can make it learnable 
        # or implement the explicit update logic in the training loop (requires exposing stats).
        # For this modular implementation, we'll make it a learnable parameter initialized to 0.
        self.expert_bias = nn.Parameter(torch.zeros(self.num_experts))

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        
        # --- Shared Experts Path ---
        # Sum of all shared experts (or concat? Usually sum or weighted sum. DeepSeek uses sum/add)
        shared_output = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_output += expert(x_flat)
            
        # --- Routed Experts Path ---
        # Logits + Bias
        router_logits = self.router(x_flat) + self.expert_bias
        
        # Top-K
        routing_weights, selected_experts = torch.topk(router_logits, self.num_experts_per_tok, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float).to(x.dtype)
        
        routed_output = torch.zeros_like(x_flat)
        
        # Iterate over k choices (Naive loop for readability, same as DynamicRouterMoE)
        for k in range(self.num_experts_per_tok):
            expert_indices = selected_experts[:, k]
            weights = routing_weights[:, k].unsqueeze(1)
            
            for i, expert in enumerate(self.routed_experts):
                token_mask = (expert_indices == i)
                if token_mask.any():
                    routed_output[token_mask] += weights[token_mask] * expert(x_flat[token_mask])

        # Combine
        final_output = shared_output + routed_output
        
        return final_output.view(B, T, C)

