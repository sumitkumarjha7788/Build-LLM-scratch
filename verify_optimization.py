import torch
import torch.nn as nn
from model_utils import ModelConfig
from model import GPTLanguageModel
from quantization import quantize_model, QuantizedLinear
from distillation import DistillationTrainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_quantization():
    print("--- Testing Quantization ---")
    config = ModelConfig(vocab_size=65, block_size=32, n_embd=64, n_head=2, n_layer=1, device=device)
    model = GPTLanguageModel(config).to(device)
    
    # Run once before quantization
    x = torch.randint(0, 65, (1, 32)).to(device)
    model(x)
    
    # Quantize
    model = quantize_model(model)
    
    # Check layer replacement
    linear_count = 0
    quant_count = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            linear_count += 1
        if isinstance(m, QuantizedLinear):
            quant_count += 1
            
    print(f"Quantized Layers found: {quant_count}")
    if quant_count > 0:
        print("Model replacement: PASSED")
    else:
        print("Model replacement: FAILED")
        
    # Run forward pass (simulated int8)
    try:
        logits, _ = model(x)
        print("Forward pass (Quantized): PASSED")
        print(f"Logits shape: {logits.shape}")
    except Exception as e:
        print(f"Forward pass (Quantized): FAILED: {e}")

def test_distillation():
    print("\n--- Testing Distillation ---")
    # Teacher (Larger)
    t_config = ModelConfig(vocab_size=65, block_size=32, n_embd=64, n_head=2, n_layer=2, device=device)
    teacher = GPTLanguageModel(t_config).to(device)
    
    # Student (Smaller)
    s_config = ModelConfig(vocab_size=65, block_size=32, n_embd=32, n_head=2, n_layer=1, device=device)
    student = GPTLanguageModel(s_config).to(device)
    
    trainer = DistillationTrainer(teacher, student, alpha=0.5, temperature=2.0)
    
    # Dummy Batch
    x = torch.randint(0, 65, (4, 32)).to(device)
    
    # Train step
    total_loss, kl_loss, ce_loss = trainer.train_step(x, x) # x is target for causal LM
    
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"KL Loss: {kl_loss:.4f}")
    print(f"CE Loss: {ce_loss:.4f}")
    
    if total_loss > 0:
        print("Distillation Step: PASSED")
    else:
        print("Distillation Step: FAILED (Low loss/Zero?)")

if __name__ == '__main__':
    test_quantization()
    test_distillation()
