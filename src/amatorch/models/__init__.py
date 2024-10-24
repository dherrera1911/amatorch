"""
Subpackage with the implemented AMA model variants.
"""

from .ama_gauss import AMAGauss

__all__ = ["AMAGauss"]


def __dir__():
    return __all__
