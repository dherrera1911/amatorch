import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches

from ..data_wrangle import statistics_dim_subset
from .utils import get_class_colors


def single_ellipse(center, covariance, ax, color="black"):
    """
    Plot an ellipse with a given center and covariance matrix.

    Parameters
    ----------
    center : torch.Tensor
        Center of the ellipse. Shape (2).
    covariance : torch.Tensor
        Covariance matrix of the ellipse. Shape (2, 2).
    ax : matplotlib.axes.Axes
        Axes to plot the ellipse.
    color : str, optional
        Color of the ellipse. The default is 'black'.
    """
    eig_val, eig_vec = torch.linalg.eigh(covariance)
    # Get the angle of the ellipse main axis
    angle = torch.atan2(eig_vec[1, 0], eig_vec[0, 0])
    # Get the length of the axes
    scale = torch.sqrt(eig_val)
    # Plot the ellipse
    ellipse = patches.Ellipse(
        xy=center,
        width=scale[0] * 4,
        height=scale[1] * 4,
        angle=angle * 180 / np.pi,
        color=color,
    )
    ellipse.set_facecolor("none")
    ellipse.set_linewidth(3)
    ax.add_patch(ellipse)


def statistics_ellipses(
    means, covariances, filter_pair=(0, 1), ax=None, values=None, color_map="viridis"
):
    """
    Plot the ellipses of the filter response statistics across classes.

    Parameters
    ----------
    means : torch.Tensor
        Means of the filter responses. Shape (n_classes, n_filters).
    covariances : torch.Tensor
        Covariances of the filter responses.
        Shape (n_classes, n_filters, n_filters).
    filter_pair : tuple of int, optional
        Pair of filters to plot. The default is [0, 1].
    ax : matplotlib.axes.Axes, optional
        Axes to plot the ellipses. If None, a new figure is created.
        The default is None.
    values : torch.Tensor, optional
        Values to color code the ellipses. Each value corresponds to a
        class. The default is linearly spaced values between -1 and 1.
    color_map : str or matplotlib.colors.Colormap, optional
        Color map to use for the ellipses. The default is 'viridis'.
    """
    n_classes = covariances.shape[0]

    if ax is None:
        fig, ax = plt.subplots()

    if isinstance(color_map, str):
        color_map = plt.get_cmap(color_map)

    if values is None:
        values = torch.linspace(-1, 1, n_classes)

    means_subset, covariances_subset = statistics_dim_subset(
        means, covariances, filter_pair
    )

    class_colors = get_class_colors(color_map, values)

    for i in range(n_classes):
        single_ellipse(
            center=means_subset[i],
            covariance=covariances_subset[i],
            ax=ax,
            color=class_colors[i],
        )
    ax.autoscale_view()
    return ax
