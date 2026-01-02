import torch
import torch.nn.functional as F
from alignment import DirectPreferenceOptimization, ChainOfThoughtTraining

def test_dpo_loss():
    print("--- Testing DPO Loss ---")
    dpo_loss = DirectPreferenceOptimization(beta=0.1)
    
    # 1. Case where Policy prefers Chosen more than Reference (Good)
    # Policy: Chosen=0.9, Rejected=0.1
    # Ref:    Chosen=0.5, Rejected=0.5
    # Logits from log_probs roughly:
    # pi_chosen = 0, pi_rejected = -2.0
    # ref_chosen = -0.69, ref_rejected = -0.69
    
    pi_chosen = torch.tensor([-0.1])
    pi_rejected = torch.tensor([-2.3]) # much lower
    ref_chosen = torch.tensor([-0.69])
    ref_rejected = torch.tensor([-0.69])
    
    loss, _, _ = dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected)
    print(f"Loss (Good Policy): {loss.item():.4f} (Should be low)")
    
    # 2. Case where Policy prefers Rejected (Bad)
    pi_chosen_bad = torch.tensor([-2.3])
    pi_rejected_bad = torch.tensor([-0.1])
    
    loss_bad, _, _ = dpo_loss(pi_chosen_bad, pi_rejected_bad, ref_chosen, ref_rejected)
    print(f"Loss (Bad Policy):  {loss_bad.item():.4f} (Should be high)")
    
    if loss_bad > loss:
        print("DPO Logic check: PASSED")
    else:
        print("DPO Logic check: FAILED")

def test_cot_masking():
    print("\n--- Testing CoT Masking ---")
    # Simulate a sequence: [Q1, Q2, SEP, R1, R2, A1]
    # Mask should be:      [ 0,  0,   0,  1,  1,  1]
    
    sep_id = 99
    input_ids = torch.tensor([
        [1, 2, 99, 4, 5, 6],    # Standard
        [99, 4, 5, 6, 7, 8],    # Starts with SEP (empty Q)
        [1, 2, 3, 4, 5, 6]      # No SEP (Should mask nothing or everything? Implementation dependent)
    ])
    
    print("Input:\n", input_ids)
    mask = ChainOfThoughtTraining.create_mask(input_ids, separation_token_id=sep_id)
    print("Generated Mask:\n", mask)
    
    # Check row 0
    if mask[0, 0] == 0 and mask[0, 2] == 0 and mask[0, 3] == 1:
        print("Row 0: PASSED")
    else:
        print("Row 0: FAILED")

if __name__ == '__main__':
    test_dpo_loss()
    test_cot_masking()
