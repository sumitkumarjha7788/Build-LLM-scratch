import torch
from config import ModelConfig, TrainingConfig
from attention import MultiHeadLatentAttention
from moe import DeepSeekMoE, DynamicRouterMoE
from model_utils import precompute_freqs_cis
from grpo_trainer import GRPOTrainer

def test_mla():
    print("Testing MultiHeadLatentAttention (MLA)...")
    config = ModelConfig(
        n_embd=128, n_head=4, q_lora_rank=64, kv_lora_rank=64,
        attention_type="mla", use_flash_attention=False
    )
    mla = MultiHeadLatentAttention(config)
    
    B, T, C = 2, 10, 128
    x = torch.randn(B, T, C)
    head_dim = C // 4
    rope_dim = head_dim // 2
    
    # Mock freqs_cis (T, rope_dim/2)
    freqs_cis = precompute_freqs_cis(rope_dim, T*2)[:T]
    
    out = mla(x, freqs_cis)
    assert out.shape == (B, T, C), f"Output shape mismatch: {out.shape}"
    print("MLA Test Passed!")

def test_deepseek_moe():
    print("Testing DeepSeekMoE...")
    config = ModelConfig(
        n_embd=128, num_experts=8, num_experts_per_tok=2,
        num_shared_experts=1, moe_type="deepseek"
    )
    moe = DeepSeekMoE(config)
    
    B, T, C = 2, 10, 128
    x = torch.randn(B, T, C)
    
    out = moe(x)
    assert out.shape == (B, T, C), f"Output shape mismatch: {out.shape}"
    
    # Check if router bias exists
    assert hasattr(moe, 'expert_bias'), "DeepSeekMoE missing bias term for aux-free load balancing"
    print("DeepSeekMoE Test Passed!")

def test_grpo_init():
    print("Testing GRPOTrainer Init...")
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
        def forward(self, x): return self.linear(x), None
    
    class MockConfig:
        class training:
            learning_rate = 1e-4
            rlvr_beta = 0.04
            group_size = 4
        class model:
            device = 'cpu'
            
    model = MockModel()
    trainer = GRPOTrainer(model, None, MockConfig(), tokenizer=None)
    assert trainer.group_size == 4
    print("GRPO Init Test Passed!")

import traceback

if __name__ == "__main__":
    try:
        test_mla()
        test_deepseek_moe()
        test_grpo_init()
        print("\nALL DEEPSEEK MODULES VERIFIED SUCCESSFULLY.")
    except Exception as e:
        print(f"\nFAILED: {e}")
        traceback.print_exc()
        exit(1)
