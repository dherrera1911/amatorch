import torch
import torch.nn as nn
import amatorch.normalization as normalization

__all__ = ['Sphere']

def __dir__():
      return __all__


# Define the sphere constraint
class Sphere(nn.Module):
    def forward(self, X):
        """Constrains the input tensor to be on the sphere.

        The norm pooled across channels is computed and used to normalize the tensor.
        -----------------
        Arguments:
        -----------------
            - X: Tensor in Euclidean space (n_filters x n_channels x n_dim)
        -----------------
        Output:
        -----------------
            - S: Tensor on Sphere (n_filters x n_channels x n_dim)
        """
        return normalization.unit_norm(X)

    def right_inverse(self, S):
        """ Function to assign to parametrization."""
        return S
