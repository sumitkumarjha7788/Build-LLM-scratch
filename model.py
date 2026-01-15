import torch
import torch.nn as nn
from torch.nn import functional as F
from model_utils import ModelConfig, precompute_freqs_cis
from attention import GroupedQueryAttention, MultiHeadLatentAttention
from moe import DynamicRouterMoE, DeepSeekMoE

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        try:
            output = self._norm(x.float()).type_as(x)
            return output * self.weight
        except RuntimeError as e:
            if "no kernel image" in str(e):
                # Fallback to CPU for RTX 5070 compatibility if CUDA kernel is missing
                device = x.device
                x_cpu = x.cpu()
                self.weight.data = self.weight.data.cpu()
                
                output = self._norm(x_cpu.float()).type_as(x_cpu)
                res = output * self.weight
                
                # Move back to original device (and weight too, to keep state consistent-ish)
                self.weight.data = self.weight.data.to(device)
                return res.to(device)
            raise e

class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.SiLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Attention Switch
        if getattr(config, 'attention_type', 'gqa') == 'mla':
            self.sa = MultiHeadLatentAttention(config)
        else:
            self.sa = GroupedQueryAttention(config)
        
        # FFN / MoE Switch
        if config.use_moe:
            if getattr(config, 'moe_type', 'standard') == 'deepseek':
                self.ffwd = DeepSeekMoE(config)
            else:
                self.ffwd = DynamicRouterMoE(config)
        else:
            self.ffwd = FeedForward(config)

            
        self.ln1 = RMSNorm(config.n_embd)
        self.ln2 = RMSNorm(config.n_embd)

    def forward(self, x, freqs_cis):
        # Pass freqs_cis to self-attention for RoPE
        x = x + self.sa(self.ln1(x), freqs_cis)
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        # Note: No absolute position embedding table anymore! RoPE handles it.
        
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        # Precompute RoPE frequencies
        # head_dim = n_embd // n_head
        head_dim = config.n_embd // config.n_head
        self.freqs_cis = precompute_freqs_cis(
            head_dim, 
            config.block_size * 2, 
            scaling_factor=config.rope_scaling_factor
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        
        # 1. Token Embeddings
        # 1. Token Embeddings
        try:
            x = self.token_embedding_table(idx) # (B,T,C)
        except RuntimeError as e:
            if "no kernel image" in str(e):
                # Fallback for RTX 50-series where Embedding kernel is missing
                # Move embedding layer to CPU if not already
                if self.token_embedding_table.weight.device.type != 'cpu':
                    self.token_embedding_table.cpu()
                
                # Compute on CPU
                x = self.token_embedding_table(idx.cpu())
                x = x.to(device) # Move back to GPU
            else:
                raise e
        
        # 2. Prepare RoPE frequencies for this sequence length
        # freqs_cis is (Max_Len, Head_Dim/2), we need (T, Head_Dim/2)
        freqs_cis = self.freqs_cis[:T].to(device)

        # 3. Transformer Blocks
        from torch.utils.checkpoint import checkpoint
        
        for block in self.blocks:
            if self.config.use_gradient_checkpointing and self.training:
                # Checkpointing trades compute for memory
                x = checkpoint(block, x, freqs_cis, use_reentrant=False)
            else:
                x = block(x, freqs_cis)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # Start a generation loop
        for _ in range(max_new_tokens):
            # Crop to block_size if needed (though RoPE extrapolates better, local window attn still limited)
            idx_cond = idx[:, -self.config.block_size:] 
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx
