import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
import time


def gaussian_log_likelihoods(points, means, covariances):
    """ Compute log-likelihood of each class given the filter responses.
    -----------------
    Arguments:
    -----------------
        - points: Points at which to evaluate the log likelihoods (n_points x n_dim)
        - means: Mean of each class (n_classes x n_dim)
        - covariances: Covariance of each class (n_classes x n_dim x n_dim)
    -----------------
    Output:
    -----------------
        - log_likelihoods: Class log likelihoods. (n_points x n_classes)
    """
    n_dim = points.shape[-1]
    # Distances from means
    distances = points.unsqueeze(1) - means.unsqueeze(0)
    # Quadratic component of log-likelihood
    quadratic_term = -0.5 * torch.einsum('ncd,cdb,ncb->nc', distances, covariances.inverse(), distances)
    # Constant term
    constant = -0.5 * n_dim * torch.log(2*torch.tensor(torch.pi)) - \
        0.5 * torch.logdet(covariances)
    # 4) Add quadratics and constants to get log-likelihood
    return quadratic_term + constant.unsqueeze(0)


def class_statistics(points, labels):
    """ Compute the mean and covariance of each class.
    -----------------
    Arguments:
    -----------------
        - points: Data points (n_points x n_dim)
        - labels: Class label of each point (n_points)
    -----------------
    Output:
    -----------------
        - means: Mean of each class (n_classes x n_dim)
        - covariances: Covariance of each class (n_classes x n_dim x n_dim)
    """
    n_classes = int(torch.max(labels) + 1)
    n_dim = points.shape[-1]
    means = torch.zeros(n_classes, n_dim)
    covariances = torch.zeros(n_classes, n_dim, n_dim)
    for i in range(n_classes):
        indices = (labels == i).nonzero().squeeze(1)
        means[i] = points[indices].mean(0)
        covariances[i] = torch.diag(points[indices])
    return {'means': means, 'covariances': covariances}


