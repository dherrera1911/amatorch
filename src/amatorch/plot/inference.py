import matplotlib.pyplot as plt
import torch

from ..data_wrangle import subsample_class_points
from .output_summary import output_statistics


def scatter_estimates(
    estimates, true_values, jitter=0.0, ax=None, points_per_class=1000
):
    """
    Plot the model estimates.

    Parameters
    ----------
    estimates : torch.Tensor
        The estimated value for each stimulus (n_stimuli).
    true_values : torch.Tensor
        The true value for each stimulus (n_stimuli).
    jitter : float
        Amount of noise to add to the true values for plotting.
    """
    if ax is None:
        fig, ax = plt.subplots()
    # Subsample the points
    subsampled_estimates, subsampled_true_values = subsample_class_points(
        estimates, true_values, points_per_class
    )
    ax.scatter(
        true_values + torch.randn(len(true_values)) * jitter,
        estimates + torch.randn(len(estimates)) * jitter,
        color="grey",
        alpha=0.2,
        s=3,
    )
    ax.plot(
        [true_values.min(), true_values.max()],
        [true_values.min(), true_values.max()],
        "k--",
        label="Identity",
    )
    ax.set_xlabel("True value")
    ax.set_ylabel("Estimated value")
    return ax


def plot_estimates_statistics(
    estimates, true_values, ax=None, ci_bars=False, quantiles=[0.025, 0.975]
):
    """
    Plot the mean estimates by true value.

    Parameters
    ----------
    estimates : torch.Tensor
        The estimated value for each stimulus (n_stimuli).
    labels : torch.int64
        The label for each stimulus (n_stimuli).
    ax : matplotlib.axes.Axes
        The axes to plot on.
    ci_bars : bool
        Whether to plot confidence intervals.
    """
    if ax is None:
        fig, ax = plt.subplots()
    # Generate labels from the true values
    x_values, labels = torch.unique(true_values, return_inverse=True)
    estimate_statistics = output_statistics(estimates, labels, quantiles=quantiles)
    ax.plot(x_values, estimate_statistics["mean"], label="Mean")
    if ci_bars:
        ax.fill_between(
            x_values,
            estimate_statistics["ci_low"],
            estimate_statistics["ci_high"],
            alpha=0.5,
        )
    ax.set_xlabel("True value")
    ax.set_ylabel("Estimated value")
    return ax
