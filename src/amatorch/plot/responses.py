import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data_wrangle import subsample_class_points
from .utils import get_class_colors


def scatter_responses(
    responses, labels, ax=None, values=None, filter_pair=(0, 1), n_points=1000,
    classes_plot=None
):
    """
    Plot scatter of the responses to different categories.

    Parameters
    ----------
    responses : torch.Tensor
        Responses to the stimuli. Shape (n_stimuli, n_filters).
    labels : torch.int64
        Class labels of each point with shape (n_points).
    ax : matplotlib.axes.Axes, optional
        Axes to plot the scatter. If None, a new figure is created.
        The default is None.
    values : torch.Tensor, optional
        Values to color the points. The default is linearly spaced values
        between -1 and 1.
    filter_pair : tuple, optional
        Pair of filters to plot. The default is (0, 1).
    n_points : int, optional
        Number of points per class to plot. The default is 1000.
    classes_plot : list, optional
        List of classes to plot. The default is all classes.
    """
    responses_plt, labels_plt = subsample_class_points(responses, labels, n_points)
    responses_plt = responses_plt[:, filter_pair].detach().cpu().numpy()
    labels_plt = labels_plt.detach().cpu().numpy()

    if values is None:
        values = np.linspace(-1, 1, len(labels.unique()))

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    if classes_plot is not None:
        inds_keep = np.isin(labels_plt, np.array(classes_plot))
        responses_plt = responses_plt[inds_keep]
        labels_plt = labels_plt[inds_keep]

    color_map = plt.get_cmap("viridis")
    class_colors = get_class_colors(color_map, values)

    ax.scatter(
        responses_plt[:, 0],
        responses_plt[:, 1],
        c=class_colors[labels_plt],
        s=10,
        alpha=0.5,
    )

    ax.set_xlabel(f"Filter {filter_pair[0] + 1}")
    ax.set_ylabel(f"Filter {filter_pair[1] + 1}")
    return ax
