import torch.nn as nn

import amatorch.normalization as normalization

__all__ = ["Sphere"]


def __dir__():
    return __all__


# Define the sphere constraint
class Sphere(nn.Module):
    """
    Constrains the input tensor to lie on the sphere.
    """

    def forward(self, X):
        """
        Normalize the input tensor so that it lies on the sphere.

        The norm pooled across channels is computed and used to normalize the tensor.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor in Euclidean space with shape (n_filters, n_channels, n_dim).

        Returns
        -------
        torch.Tensor
            Normalized tensor lying on the sphere with shape
            (n_filters, n_channels, n_dim).
        """
        return normalization.unit_norm(X)

    def right_inverse(self, S):
        """
        Identity function to assign to parametrization.

        Parameters
        ----------
        S : torch.Tensor
            Input tensor. Should be different from zero.

        Returns
        -------
        torch.Tensor
            Returns the input tensor `S`.
        """
        return S
