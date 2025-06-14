import torch
import torch.nn as nn

class RowNormActivation(nn.Module):
    """
    Activation function that normalizes each row vector of the input matrix.
    For each row vector x, it computes x / ||x|| where ||x|| is the L2 norm of x.
    """
    def __init__(self, eps=1e-8):
        super(RowNormActivation, self).__init__()
        self.eps = eps  # Small constant to prevent division by zero
    
    def forward(self, x):
        # Calculate L2 norm for each row
        # Keep dimensions for broadcasting
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        # Add small epsilon to prevent division by zero
        norm = norm + self.eps
        # Normalize each row
        return x / norm