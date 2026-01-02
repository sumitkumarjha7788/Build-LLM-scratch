import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationTrainer:
    """
    Teacher-Student Knowledge Distillation Trainer.
    Loss = alpha * SoftTarget(KL) + (1-alpha) * HardTarget(CE)
    """
    def __init__(self, teacher_model, student_model, alpha=0.5, temperature=2.0):
        self.teacher = teacher_model
        self.student = student_model
        self.alpha = alpha
        self.T = temperature
        
        # Teacher is usually frozen
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()
        
    def train_step(self, x, targets):
        """
        Performs one distillation step.
        """
        # 1. Teacher Forward (No Grad)
        with torch.no_grad():
            teacher_logits, _ = self.teacher(x)
            # Flatten to match student if student flattens (which it does when targets provided)
            B, T, C = teacher_logits.shape
            teacher_logits = teacher_logits.view(B*T, C)
            
        # 2. Student Forward
        student_logits, student_loss_ce = self.student(x, targets) 
        # Note: student_loss_ce is the Standard Cross Entropy (Hard Target)
        
        # 3. Soft Target Loss (KL Divergence)
        # LogSoftmax(Student/T) vs Softmax(Teacher/T)
        
        soft_targets = F.softmax(teacher_logits / self.T, dim=-1)
        student_log_soft = F.log_softmax(student_logits / self.T, dim=-1)
        
        # KL Div: input=log_probs, target=probs
        # reduction='batchmean' aligns with math definition
        loss_kl = F.kl_div(student_log_soft, soft_targets, reduction='batchmean') * (self.T**2)
        
        # 4. Combined Loss
        total_loss = self.alpha * loss_kl + (1 - self.alpha) * student_loss_ce
        
        return total_loss, loss_kl.item(), student_loss_ce.item()
