# LLM Integration Demo Guide

This guide shows you how to use the unified LLM training and optimization system.

## 🚀 Quick Start

### 1. Training a Model
Train a new model using the unified entry point.

```bash
# Train on TinyShakespeare (Small)
python main.py --mode train --dataset tinyshakespeare --iters 1000

# Train on TinyStories (Large - Requires prepare_dataset.py first)
python main.py --mode train --dataset tinystories --iters 5000
```

### 2. Text Generation (Inference)
Generate text using a saved checkpoint.

```bash
# Generate from the last saved model
python main.py --mode generate --resume checkpoints/last_model.pt --prompt "To be or not to be"
```

### 3. Verification
Run the comprehensive suite to verify all architectural components.

```bash
# End-to-end integration test
python main.py --mode train --iters 10 --dataset tinyshakespeare

# Verify specific features
python verify_architecture.py   # RoPE & GQA
python verify_moe.py            # Mixture of Experts
python verify_alignment.py      # SFT & DPO logic
```

---

## 🛠️ Advanced Features

### Mixture of Experts (MoE)
Toggle MoE in `config.py`:
```python
# config.py
model = ModelConfig(
    use_moe=True,
    num_experts=8,
    num_experts_per_tok=2
)
```

### Optimization
Apply pruning and quantization in `inference.py` or through the `InferenceEngine` interface to reduce model size for deployment.

### RAG & Speculative Decoding
The `InferenceEngine` in `inference.py` supports:
- **RAG**: Pass a list of documents to enable adaptive retrieval.
- **Speculative**: Pass a `draft_model` to speed up generation.

---

## 📈 Monitoring
- **Logs**: Check `training_log.csv` for step-by-step loss metrics.
- **Plots**: View `plots/training_loss.png` for a visual update on training progress.
- **Checkpoints**: Best and latest models are always kept in the `checkpoints/` folder.
