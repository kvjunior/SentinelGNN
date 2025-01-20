import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def to_var(x, needs_cuda, volatile=False):
    """Convert a tensor to a PyTorch variable with optional CUDA support."""
    if torch.cuda.is_available() and needs_cuda:
        x = x.cuda()
    return Variable(x, volatile=volatile)

def mkdir(directory):
    """Create a directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def calculate_adaptive_loss(sample_energy, clip_loss, diversity_loss, entropy_loss, lambda_energy, lambda_recon, lambda_diversity, lambda_entropy):
    """Calculate the adaptive loss based on various components."""
    loss = lambda_energy * torch.mean(sample_energy) + lambda_recon * clip_loss 
    loss += lambda_diversity * diversity_loss + lambda_entropy * entropy_loss
    return loss

def log_sigmoid(x):
    """Compute the log sigmoid of input tensor."""
    return torch.clamp(x, max=0) - torch.log(torch.exp(-torch.abs(x)) + 1)