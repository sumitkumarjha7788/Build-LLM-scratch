import torch
import torch.nn as nn

class MagnitudePruner:
    """
    Implements Magnitude-based weight pruning for Linear layers.
    """
    @staticmethod
    def prune_model(model: nn.Module, amount: float = 0.3):
        """
        Prunes the specified amount of weights with the smallest L1-norm.
        
        Args:
            model: The GPT model to prune.
            amount: Fraction of weights to prune (0.0 to 1.0).
        """
        print(f"Pruning {amount*100:.1f}% of weights...")
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Calculate number of weights to prune
                tensor = module.weight.data
                num_weights = tensor.numel()
                num_to_prune = int(num_weights * amount)
                
                if num_to_prune > 0:
                    # Compute L1 norm (absolute value)
                    abs_weights = tensor.abs()
                    
                    # Find threshold
                    # We pick the value that is at the num_to_prune position in flattened sorted list
                    threshold = torch.kthvalue(abs_weights.view(-1), num_to_prune).values
                    
                    # Create mask
                    mask = abs_weights > threshold
                    
                    # Apply mask
                    module.weight.data *= mask.float()
                    
                    print(f"Pruned layer: {name}. Sparsity: {(~mask).float().mean():.4f}")
        
        return model
