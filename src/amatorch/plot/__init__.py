"""
Utilities for plotting model parameters (filters and statistics) and
model outputs (responses, posteriors and estimates).
"""

from .filters import plot_filters
from .inference import plot_estimates_statistics, scatter_estimates
from .responses import scatter_responses
from .statistics import statistics_ellipses
from .colors import draw_color_bar

__all__ = [
    "plot_filters",
    "scatter_responses",
    "statistics_ellipses",
    "scatter_estimates",
    "plot_estimates_statistics",
    "draw_color_bar",
]


def __dir__():
    return __all__
