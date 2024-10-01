from importlib import resources
from .load import load_data

__all__ = ['disparity_data']

def __dir__():
      return __all__

# Reference the resources inside the `data` directory of the package
FILES = resources.files(__name__)


def disparity_data():
    """
    Load data from the 'disparity' subdirectory.

    Returns:
    - dict: A dictionary containing the loaded stimuli, labels, and values.
    """
    data_dir = FILES / 'disparity'
    data = load_data(data_dir=data_dir)
    n_channels = 2
    n_pixels = 26
    data['stimuli'] = data['stimuli'].reshape(-1, n_channels, n_pixels)
    return data


