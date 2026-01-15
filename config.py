import torch
from dataclasses import dataclass, field
from typing import Optional, List

def check_device():
    if not torch.cuda.is_available():
        return 'cpu'
    
    try:
        # Try a simple small tensor operation to verify CUDA is functional
        # detected sm_120 (RTX 5070) often fails with current torch stable
        x = torch.tensor([1.0], device='cuda')
        y = x * 2.0
        return 'cuda'
    except RuntimeError:
        print("Warning: CUDA is available but non-functional (likely architecture mismatch). Falling back to CPU.")
        return 'cpu'

@dataclass
class ModelConfig:
    # Architecture
    vocab_size: int = 65
    block_size: int = 256
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2
    
    # GQA
    n_kv_head: Optional[int] = None # Defaults to n_head in __post_init__
    
    # Efficiency
    use_flash_attention: bool = True
    
    # MoE
    use_moe: bool = False
    moe_type: str = "standard" # "standard" or "deepseek"
    num_experts: int = 8
    num_experts_per_tok: int = 2
    num_shared_experts: int = 1 # For DeepSeekMoE
    
    # Attention
    attention_type: str = "gqa" # "gqa", "mla"
    q_lora_rank: Optional[int] = None # For MLA compression
    kv_lora_rank: int = 512 # For MLA compression
    
    # R1 / Reasoning
    is_reasoning_model: bool = False
    max_reasoning_steps: int = 2048
    
    # Scaling
    rope_scaling_factor: float = 1.0
    
    # Device
    device: str = field(default_factory=check_device)

    def __post_init__(self):
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"

@dataclass
class TrainingConfig:
    # Basic Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    weight_decay: float = 0.1
    
    # Memory Optimizations
    gradient_accumulation_steps: int = 1
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "bf16" # "fp16" or "bf16"
    use_gradient_checkpointing: bool = False
    
    # SFT / Alignment
    sft_epochs: int = 2
    sft_learning_rate: float = 1e-5
    dpo_beta: float = 0.1
    rlvr_beta: float = 0.1
    K_samples: int = 4
    
    # Distillation
    distill_alpha: float = 0.5
    distill_temperature: float = 2.0
    
    # Checkpointing & Logging
    checkpoint_dir: str = "checkpoints"
    log_file: str = "training_log.csv"
    plot_dir: str = "plots"

@dataclass
class OptimizationConfig:
    # Pruning
    prune_amount: float = 0.3
    
    # Quantization
    use_quantization: bool = False # Whether to use quantized layers for inference

@dataclass
class GlobalConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    dataset_name: str = "tinystories" # "tinyshakespeare" or "tinystories"
    seed: int = 1337
