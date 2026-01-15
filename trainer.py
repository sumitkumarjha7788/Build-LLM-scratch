import torch
import torch.nn as nn
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

class Trainer:
    def __init__(self, model, config, optimizer, train_loader, val_loader=None, scheduler=None):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.device = config.model.device
        
        self.start_iter = 0
        self.best_val_loss = float('inf')
        
        # Prepare directories
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        os.makedirs(config.training.plot_dir, exist_ok=True)

    def train(self):
        print(f"Starting training on {self.device}...")
        self.model.train()
        
        log_file = self.config.training.log_file
        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                f.write("step,train_loss,val_loss\n")

        start_time = time.time()
        iter_count = self.start_iter
        
        # We'll use the dataloader in a loop style similar to Karpathy's nanogpt
        # but adaptable for standard PyTorch DataLoaders
        train_iter = iter(self.train_loader)

        # Helper for infinite loading
        def get_batch():
            nonlocal train_iter
            try:
                return next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                try:
                    return next(train_iter)
                except StopIteration:
                     # This implies dataset is completely empty or empty
                     raise RuntimeError("Dataset is empty or failed to load. Please check data source.")

        for i in range(self.start_iter, self.config.training.max_iters):
            xb, yb = get_batch()

            xb, yb = xb.to(self.device), yb.to(self.device)

            # Forward & Backward
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()

            # Evaluation
            if i % self.config.training.eval_interval == 0 or i == self.config.training.max_iters - 1:
                val_loss = self.evaluate()
                print(f"step {i}: train loss {loss.item():.4f}, val loss {val_loss:.4f}")
                
                # Log
                with open(log_file, "a") as f:
                    f.write(f"{i},{loss.item():.4f},{val_loss:.4f}\n")
                
                # Checkpointing
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(f"best_model.pt", i)
                
                self.save_checkpoint("last_model.pt", i)

        print(f"Training finished in {(time.time() - start_time)/60:.2f} minutes.")
        self.plot_progress()

    @torch.no_grad()
    def evaluate(self):
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        losses = []
        val_iter = iter(self.val_loader)
        
        for _ in range(self.config.training.eval_iters):
            try:
                xb, yb = next(val_iter)
            except StopIteration:
                val_iter = iter(self.val_loader)
                xb, yb = next(val_iter)
                
            xb, yb = xb.to(self.device), yb.to(self.device)
            _, loss = self.model(xb, yb)
            losses.append(loss.item())
            
        self.model.train()
        return sum(losses) / len(losses)

    def save_checkpoint(self, name, iter):
        path = os.path.join(self.config.training.checkpoint_dir, name)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'iter': iter,
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_iter = checkpoint['iter']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from {path} at step {self.start_iter}")

    def plot_progress(self):
        try:
            df = pd.read_csv(self.config.training.log_file)
            plt.figure(figsize=(10, 6))
            plt.plot(df['step'], df['train_loss'], label='Train Loss')
            plt.plot(df['step'], df['val_loss'], label='Val Loss')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(self.config.training.plot_dir, "training_loss.png")
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
        except Exception as e:
            print(f"Could not save plot: {e}")
