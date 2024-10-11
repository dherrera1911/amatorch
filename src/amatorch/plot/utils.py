import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def get_class_colors(color_map, values, color_limits=None):
    """
    Get the color for each class based on the colormap and the values.

    Parameters
    ----------
    color_map : str or matplotlib.colors.Colormap
        The colormap to use.
    values : list or numpy.ndarray or torch.Tensor
        The value of each class to be used for the color scale
    color_limits : list, optional
        The minimum and maximum values for the color scale. If not provided,
        the minimum and maximum values of `values` will be used.

    Returns
    -------
    colors : numpy.ndarray (n_classes, 4)
        The color for each class.
    """
    values = np.array(values)
    if isinstance(color_map, str):
        color_map = plt.get_cmap(color_map)
    if color_limits is None:
        color_limits = [np.min(values), np.max(values)]
    color_normalizer = Normalize(vmin=color_limits[0], vmax=color_limits[1])
    class_colors = color_map(color_normalizer(values))
    return class_colors
