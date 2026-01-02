import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantizedLinear(nn.Module):
    """
    Simulated Quantized Linear Layer (Int8).
    Replaces float32 weights with int8 weights + scale.
    """
    def __init__(self, original_layer: nn.Linear):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        # Quantize weights
        w_fp32 = original_layer.weight.data
        
        # Simple Min-Max Quantization to Int8
        # Range: [-127, 127]
        w_max = w_fp32.abs().max()
        self.scale = w_max / 127.0
        
        # Quantize: w_int8 = clip(round(w_fp32 / scale))
        w_int8 = (w_fp32 / self.scale).round().clamp(-127, 127).to(torch.int8)
        
        # Store as buffer (not parameter, so optimizer ignores it if valid)
        self.register_buffer('weight_int8', w_int8)
        
        # Bias (kept in float for simplicity in "fake" quantization schemes often)
        if original_layer.bias is not None:
            self.register_buffer('bias', original_layer.bias.data)
        else:
            self.register_buffer('bias', None)
            
    def forward(self, x):
        # Dequantize for prompt computation (Fake Quantization)
        # In real kernels, you'd do gemm_int8(x_int8, w_int8)
        
        w_fp32_approx = self.weight_int8.float() * self.scale
        
        return F.linear(x, w_fp32_approx, self.bias)

def quantize_model(model: nn.Module):
    """
    Recursively replace nn.Linear with QuantizedLinear.
    """
    print("Quantizing model layers...")
    
    # We need to traverse model carefully.
    # Simple recursion
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace
            quantized = QuantizedLinear(module)
            setattr(model, name, quantized)
            print(f"Quantized: {name}")
        else:
            # Recurse
            quantize_model(module)
            
    return model
