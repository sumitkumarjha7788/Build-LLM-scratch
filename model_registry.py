from typing import Optional
import torch.nn as nn
from config import ModelConfig

# We will move the actual model building logic here to avoid circular imports 
# or massive "if/else" chains in the main model file.

class ModelRegistry:
    _builders = {}

    @classmethod
    def register(cls, name):
        def decorator(builder_fn):
            cls._builders[name] = builder_fn
            return builder_fn
        return decorator

    @classmethod
    def build_model(cls, config: ModelConfig) -> nn.Module:
        model_type = getattr(config, 'model_type', 'baseline')
        
        if model_type not in cls._builders:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls._builders.keys())}")
        
        return cls._builders[model_type](config)

@ModelRegistry.register("baseline")
def build_baseline(config: ModelConfig) -> nn.Module:
    from model import GPTLanguageModel
    # Ensure config matches V1 expectations
    config.use_moe = False
    config.attention_type = "gqa"
    return GPTLanguageModel(config)

@ModelRegistry.register("deepseek_v3")
def build_deepseek_v3(config: ModelConfig) -> nn.Module:
    from model import GPTLanguageModel
    # Force V3 params if not set, or trust config
    config.use_moe = True
    config.moe_type = "deepseek"
    config.attention_type = "mla"
    return GPTLanguageModel(config)

@ModelRegistry.register("deepseek_r1")
def build_deepseek_r1(config: ModelConfig) -> nn.Module:
    # R1 is V3 architecture + Reasoning (GRPO) training
    # Structure is same as V3
    return build_deepseek_v3(config)
