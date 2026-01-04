import torch
import torch.optim as optim
from model import GPTLanguageModel
from model_utils import ModelConfig
from rlvr import RLVRTrainer

def test_rlvr():
    print("--- Testing RLVR Trainer ---")
    device = 'cpu'
    config = ModelConfig(
        vocab_size=65, 
        block_size=32, 
        n_embd=64, 
        n_head=2, 
        n_layer=1, 
        device=device
    )
    model = GPTLanguageModel(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    trainer = RLVRTrainer(model, optimizer, beta=0.1)
    
    # Mock Batch
    # Prompts: (B, T)
    prompts = torch.randint(0, 65, (2, 8)).to(device)
    # Ground truths can be anything for this mock test
    ground_truths = ["4", "Shakespeare"]
    
    print("Running RLVR training step...")
    try:
        loss = trainer.train_step(prompts, ground_truths, K=2)
        print(f"RLVR Step Loss: {loss:.4f}")
        
        if loss != 0:
            print("RLVR: PASSED")
        else:
            print("RLVR: FAILED (Loss is zero, check advantage logic)")
            
    except Exception as e:
        print(f"RLVR Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    test_rlvr()
