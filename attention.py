import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional
from model_utils import ModelConfig, apply_rotary_emb
import math

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
            try:
                output = F.scaled_dot_product_attention(
                    xq, xk, xv,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=True
                )
            except RuntimeError:
                # Fallback to manual implementation if kernel missing (common on new GPUs)
                # (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
                scale = (1.0 / math.sqrt(self.head_dim))
                att = (xq @ xk.transpose(-2, -1)) * scale
                
                # Causal mask
                mask = torch.tril(torch.ones(T, T, device=xq.device)).view(1, 1, T, T)
                att = att.masked_fill(mask == 0, float('-inf'))
                
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                output = att @ xv
        else:
            # Manual fallback
            scale = (1.0 / math.sqrt(self.head_dim))
            att = (xq @ xk.transpose(-2, -1)) * scale
            mask = torch.tril(torch.ones(T, T, device=xq.device)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            output = att @ xv

        # 7. Reshape back: (B, H, T, D) -> (B, T, H, D) -> (B, T, C)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        
        # 8. Output projection
        return self.resid_dropout(self.wo(output))

class MultiHeadLatentAttention(nn.Module):
    """
    DeepSeek-V3 Multi-Head Latent Attention (MLA).
    Key features:
    1. Low-rank KV compression to reduce KV cache size.
    2. Decoupled RoPE: Position info acts on a separate part of the vector.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank if config.q_lora_rank is not None else config.n_embd
        
        # Derived dimensions
        # DeepSeek V3 splits headers into 'pe' (positional) and 'nope' (non-positional/content)
        # For simplicity in this implementation, we will assume:
        # - The latent KV generates the 'content' part of K and all of V.
        # - A separate path (or split) handles the 'rope' part of K.
        # However, to fit into the standard Block structure where we just return 'y', 
        # we'll implement the compression logic that runs during training.
        
        # 1. Query Compression (if q_lora_rank < n_embd, otherwise identity-ish)
        self.w_dq = nn.Linear(self.n_embd, self.q_lora_rank, bias=False)
        self.w_uq = nn.Linear(self.q_lora_rank, self.n_head * self.head_dim, bias=False)
        
        # 2. Key-Value Compression (The core MLE part)
        # Compresses input into a latent vector c_KV
        self.w_dkv = nn.Linear(self.n_embd, self.kv_lora_rank, bias=False)
        
        # Up-projects c_KV to generate Key (content part) and Value
        self.w_uk = nn.Linear(self.kv_lora_rank, self.n_head * self.head_dim, bias=False)
        self.w_uv = nn.Linear(self.kv_lora_rank, self.n_head * self.head_dim, bias=False)
        
        # 3. Decoupled RoPE for Keys
        # DeepSeek adds a separate key that carries strictly RoPE info
        # We'll use a simplified version: simpler learnable query/key rope generators
        # or just project input x directly to a small 'rope' dimension.
        # Let's strictly follow the V3 idea:
        # k_rope is derived from x directly (or via w_dkv? Paper says decoupled)
        # Actually usually it is k_rope = w_kr(x).
        self.rope_dim = self.head_dim // 2 # Arbitrary split for this impl, often usage dependent
        self.w_qr = nn.Linear(self.q_lora_rank, self.n_head * self.rope_dim, bias=False)
        self.w_kr = nn.Linear(self.n_embd, self.n_head * self.rope_dim, bias=False) # K_rope depends on x, not c_KV usually
        
        self.wo = nn.Linear(self.n_head * self.head_dim, self.n_embd, bias=False)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, freqs_cis):
        B, T, C = x.shape
        
        # 1. Compressed Latent KV
        c_kv = self.w_dkv(x) # (B, T, kv_lora_rank)
        
        # 2. Generate Content K and V from latent
        # (B, T, n_head * head_dim)
        k_content = self.w_uk(c_kv)
        v = self.w_uv(c_kv) # V is fully derived from latent
        
        # 3. Compressed Query
        c_q = self.w_dq(x) # (B, T, q_lora_rank)
        q_content = self.w_uq(c_q) # (B, T, n_head * head_dim)
        
        # 4. RoPE Parts
        # q_rope: (B, T, n_head * rope_dim)
        # k_rope: (B, T, n_head * rope_dim)
        q_rope = self.w_qr(c_q)
        k_rope = self.w_kr(x) # Driven by input directly
        
        # Reshape for heads
        q_content = q_content.view(B, T, self.n_head, self.head_dim)
        k_content = k_content.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        
        q_rope = q_rope.view(B, T, self.n_head, self.rope_dim)
        k_rope = k_rope.view(B, T, self.n_head, self.rope_dim)
        
        # Apply RoPE to the rope parts ONLY
        # apply_rotary_emb expects (B, T, H, D)
        # Note: freqs_cis must match rope_dim. Our current precompute handles full head_dim.
        # We need to slice freqs_cis to rope_dim.
        freqs_cis_sliced = freqs_cis[..., :self.rope_dim // 2] # complex is half dim
        
        q_rope, k_rope = apply_rotary_emb(q_rope, k_rope, freqs_cis_sliced)
        
        # 5. Concatenate Content + RoPE for Q and K?
        # DeepSeek V3 paper uses a concat strategy for attention scoring.
        # score = (q_c^T k_c) + (q_r^T k_r) typically.
        # Which is equivalent to concat(q_c, q_r) @ concat(k_c, k_r).T
        # BUT dimensions must match.
        # q_content is head_dim. q_rope is rope_dim.
        # Wait, usually the implementation makes head_dim = content_dim + rope_dim (or checks out).
        # We'll allow them to be separate but we need to compute scores.
        # Actually V3 allows standard FA if we concat them into a single vector.
        # So we construct:
        # Q = [q_content, q_rope]
        # K = [k_content, k_rope]
        # V = [v]       (V has no rope part)
        
        # Note on dims:
        # If we want to use FlashAttn, Q and K must have same dimension D.
        # Here D_q = head_dim + rope_dim.
        # We defined self.head_dim. Let's assume the linear layers output `head_dim` meant "content dim".
        # So effective head size for attention is head_dim + rope_dim.
        
        q = torch.cat([q_content, q_rope], dim=-1) # (B, T, H, head_dim + rope_dim)
        k = torch.cat([k_content, k_rope], dim=-1)
        
        # V has normal head_dim. 
        # SDPA requires Q, K, V last dim to be same? No, V can be different D_v in standard attention,
        # but PyTorch F.scaled_dot_product_attention usually assumes D_k == D_v?
        # Actually standard Attention: Q(L, D_k), K(S, D_k), V(S, D_v).
        # So it IS allowed.
        
        # Reshape for efficient attention: (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Flash Attention
        # Flash Attention
        try:
            output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True
            )
        except RuntimeError:
            # Fallback
            D = q.shape[-1]
            scale = 1.0 / math.sqrt(D)
            att = (q @ k.transpose(-2, -1)) * scale
            
            mask = torch.tril(torch.ones(T, T, device=q.device)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            output = att @ v
        
        # (B, H, T, D_v) -> (B, T, H, D_v) -> (B, T, C)
        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        
        return self.resid_dropout(self.wo(output))
