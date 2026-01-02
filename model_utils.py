import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ModelConfig:
    vocab_size: int = 65
    block_size: int = 256
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2
    n_kv_head: Optional[int] = None # For GQA. If None, defaults to n_head (MHA)
    use_flash_attention: bool = True
    use_moe: bool = False
    num_experts: int = 8
    num_experts_per_tok: int = 2
    rope_scaling_factor: float = 1.0 # Linear scaling factor for context extension
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __post_init__(self):
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, scaling_factor: float = 1.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    Supports Linear RoPE Scaling via scaling_factor.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # Linear Scaling: Scale the time indices
    # t_new = t / scaling_factor
    t = torch.arange(end, device=freqs.device) / scaling_factor 
    
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Broadcast frequencies to match batch and head dimensions
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)
