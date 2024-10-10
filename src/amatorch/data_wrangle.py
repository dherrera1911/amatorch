import torch

__all__ = ["statistics_dim_subset"]


def __dir__():
    return __all__


def statistics_dim_subset(means, covariances, keep_inds):
    """
    Keep the statistics for a subset of the dimensions. 

    Parameters
    ----------
    means : torch.Tensor
        Means of the data. (n_classes, n_dim)
    covariances : torch.Tensor
        Covariances of the data. (n_classes, n_dim, n_dim) 
    keep_inds : torch.Tensor
        Indices of the dimensions to keep. (n_keep_dim,)

    Returns
    -------
    means_subset : torch.Tensor
        Means of the data for the subset of dimensions. (n_classes, n_keep_dim)
    covariances_subset : torch.Tensor
        Covariances of the data for the subset of dimensions.
        (n_classes, n_keep_dim, n_keep_dim)
    """
    means_subset = means[:, keep_inds]
    covariances_subset = covariances[:, keep_inds][:, :, keep_inds]
    return means_subset, covariances_subset
