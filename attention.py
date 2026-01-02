import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional
from model_utils import ModelConfig, apply_rotary_emb

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = self.n_head // self.n_kv_head

        # Weights
        self.wq = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout_p = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        B, T, C = x.shape
        
        # 1. Project Q, K, V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        # 2. Reshape for heads: (B, T, H, D) -> (B, H, T, D) transposing later in SDPA? 
        # Actually RoPE needs (B, T, H, D) usually. 
        xq = xq.view(B, T, self.n_head, self.head_dim)
        xk = xk.view(B, T, self.n_kv_head, self.head_dim)
        xv = xv.view(B, T, self.n_kv_head, self.head_dim)

        # 3. Apply RoPE (Rotary Position Embeddings)
        # Note: apply_rotary_emb expects complex numbers if doing complex mult,
        # but our helper handles the view conversions.
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # 4. Repeat K/V heads for GQA (if n_kv_head < n_head)
        # (B, T, n_kv_head, D) -> (B, T, n_head, D)
        if self.n_rep > 1:
            xk = xk.unsqueeze(3).repeat(1, 1, 1, self.n_rep, 1).flatten(2, 3)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, self.n_rep, 1).flatten(2, 3)

        # 5. Prepare for scaled_dot_product_attention: (B, H, T, D)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 6. Flash Attention (Scaled Dot Product Attention)
        # is_causal=True automatically handles the triangular mask
        if self.config.use_flash_attention:
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True
            )
        else:
            # Manual fallback (for debugging or old pytorch)
            # Not fully implementing fallback here to enforce Flash use as per plan
             output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True
            )

        # 7. Reshape back: (B, H, T, D) -> (B, T, H, D) -> (B, T, C)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        
        # 8. Output projection
        return self.resid_dropout(self.wo(output))
