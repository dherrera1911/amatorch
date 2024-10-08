from importlib import resources

from .load import load_data, load_filters

__all__ = ["disparity_data", "disparity_filters"]


def __dir__():
    return __all__


# Reference the resources inside the `data` directory of the package
FILES = resources.files(__name__)


def disparity_data():
    """
    Load data from the 'disparity' subdirectory.

    Returns
    -------
    dict
        A dictionary containing the loaded stimuli, labels, and values.
        The 'stimuli' are reshaped to (n_samples, n_channels, n_pixels),
        where `n_channels=2` and `n_pixels=26`.
    """
    data_dir = FILES / "disparity"
    data = load_data(data_dir=data_dir)
    n_channels = 2
    n_pixels = 26
    data["stimuli"] = data["stimuli"].reshape(-1, n_channels, n_pixels)
    return data


def disparity_filters():
    """
    Load filters from the 'disparity' subdirectory.

    Returns
    -------
    torch.Tensor
        A tensor of shape (n_filters, n_channels, n_pixels) containing the filters,
        where `n_channels=2` and `n_pixels=26`.
    """
    data_dir = FILES / "disparity"
    filters = load_filters(data_dir=data_dir)
    n_channels = 2
    n_pixels = 26
    filters = filters.reshape(-1, n_channels, n_pixels)
    return filters
