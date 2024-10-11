import torch
import numpy as np

__all__ = ["statistics_dim_subset", "subsample_class_points"]


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


def subsample_class_points(points, labels, n_per_class):
    """
    Return a subsample of points for each class (whichever is
    smaller of n_per_class or the total points in the class).

    Parameters
    ----------
    points : torch.Tensor
        Points to subsample. (n_points, n_dim)
    labels : torch.Tensor
        Labels or values of the points. (n_points,)
    n_per_class : int
        Number of points to subsample for each class.

    Returns
    -------
    subsampled_points : torch.Tensor
        Subsampled points. (n_classes * n_points_per_class, n_dim)
    subsampled_labels : torch.Tensor
        Labels or values of the subsampled points. (n_classes * n_points_per_class,)
    """
    n_classes = np.unique(labels)
    subsampled_points = []
    subsampled_labels = []

    for i, label in enumerate(labels.unique()):
        class_points = points[labels == label]

        n_points_class = np.min([n_per_class, class_points.shape[0]])
        subsampled_points.append(class_points[:n_points_class])
        subsampled_labels.append(label.repeat(n_points_class))

    subsampled_points = torch.cat(subsampled_points, dim=0)
    subsampled_labels = torch.cat(subsampled_labels, dim=0)
    return subsampled_points, subsampled_labels
