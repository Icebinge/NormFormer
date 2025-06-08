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

# Example usage:
if __name__ == "__main__":
    # Create a sample input tensor
    x = torch.tensor([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0],
                     [7.0, 8.0, 9.0]], dtype=torch.float32)
    
    # Initialize the activation function
    row_norm = RowNormActivation()
    
    # Apply the activation
    output = row_norm(x)
    
    print("Input tensor:")
    print(x)
    print("\nNormalized output:")
    print(output)
    
    # Verify that each row has unit norm
    row_norms = torch.norm(output, p=2, dim=1)
    print("\nNorms of each row in output:")
    print(row_norms)
