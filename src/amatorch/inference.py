import torch

__all__ = ["gaussian_log_likelihoods", "class_statistics"]


def __dir__():
    return __all__


def gaussian_log_likelihoods(points, means, covariances):
    """
    Compute the log-likelihood of each class assuming conditional
    Gaussian distributions.

    Parameters
    ----------
    points : torch.Tensor
        Points at which to evaluate the log-likelihoods with shape (n_points, n_dim).
    means : torch.Tensor
        Mean of each class with shape (n_classes, n_dim).
    covariances : torch.Tensor
        Covariance matrix of each class with shape (n_classes, n_dim, n_dim).

    Returns
    -------
    log_likelihoods: torch.Tensor
        Log-likelihoods for each class with shape (n_points, n_classes).
    """
    n_dim = points.shape[-1]
    # Distances from means
    distances = points.unsqueeze(-2) - means.unsqueeze(0)
    # Quadratic component of log-likelihood
    quadratic_term = -0.5 * torch.einsum(
        "...cd,cdb,...cb->...c", distances, covariances.inverse(), distances
    )
    # Constant term
    constant = -0.5 * n_dim * torch.log(
        2 * torch.tensor(torch.pi)
    ) - 0.5 * torch.logdet(covariances)
    # 4) Add quadratics and constants to get log-likelihood
    log_likelihoods = quadratic_term + constant.unsqueeze(0)
    return log_likelihoods.squeeze()


def class_statistics(points, labels):
    """
    Compute the mean and covariance of each class.

    Parameters
    ----------
    points : torch.Tensor
        Data points with shape (n_points, n_dim).
    labels : torch.Tensor
        Class labels of each point with shape (n_points).

    Returns
    -------
    dict
        A dictionary containing:
        - means: torch.Tensor of shape (n_classes, n_dim), the mean of each class.
        - covariances: torch.Tensor of shape (n_classes, n_dim, n_dim), the
            covariance matrix of each class.
    """
    n_classes = int(torch.max(labels) + 1)
    n_dim = points.shape[-1]
    means = torch.zeros(n_classes, n_dim)
    covariances = torch.zeros(n_classes, n_dim, n_dim)
    for i in range(n_classes):
        indices = (labels == i).nonzero().squeeze(1)
        means[i] = torch.mean(points[indices], dim=0)
        covariances[i] = torch.cov(points[indices].t())
    return {"means": means, "covariances": covariances}
