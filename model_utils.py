import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from config import ModelConfig

class SimpleTokenizer:
    """ A simple character-level tokenizer. """
    def __init__(self, chars: Optional[List[str]] = None, text: Optional[str] = None):
        if chars:
            self.chars = sorted(list(set(chars)))
        elif text:
            self.chars = sorted(list(set(text)))
        else:
            self.chars = []
        
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
        
    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s if c in self.stoi]
        
    def decode(self, l: List[int]) -> str:
        return ''.join([self.itos[i] for i in l if i in self.itos])

    def save_vocab(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            f.write("".join(self.chars))
            
    @classmethod
    def load_vocab(cls, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            chars = list(f.read())
        return cls(chars=chars)

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
