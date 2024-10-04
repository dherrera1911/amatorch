import torch

__all__ = ['unit_norm', 'unit_norm_channels']

def __dir__():
      return __all__


def unit_norm(stimuli, c50=torch.as_tensor(0)):
    """Normalize stimuli to have norm less or equal to 1.

    Channels are normalized by their aggregated offsetted norm
    (i.e. || sum of squares + c50 ||).
    ----------------
    Arguments:
    ----------------
      - stimuli: Stimuli tensor (n_stim x n_channels x n_dim)
    ----------------
    Outputs:
    ----------------
      - stimuli_normalized: Normalized stimuli. (nStim x nDim)
    """
    # Normalizing factor
    normalizing_factor = torch.sqrt(torch.sum(stimuli**2, dim=(-2,-1)) + c50)
    return stimuli / normalizing_factor[:, None, None]


def unit_norm_channels(stimuli, c50=torch.as_tensor(0)):
    """Normalize stimuli to have norm less or equal to 1,
    normalizing each channel separately.

    Each channel is normalized separately by its offsetted
    norm (i.e. sqrt( sum of squares + c50)), and both channels
    are normalized by the square root of the number of channels.
    ----------------
    Arguments:
    ----------------
      - stimuli: Stimuli tensor (n_stim x n_channels x n_dim)
    ----------------
    Outputs:
    ----------------
      - stimuli_normalized: Normalized stimuli. (nStim x nDim)
    """
    # Normalizing factor
    n_channels = torch.as_tensor(stimuli.shape[1], dtype=stimuli.dtype, device=stimuli.device)
    normalizing_factor = torch.sqrt(torch.sum(stimuli**2, dim=-1) + c50) * torch.sqrt(n_channels)
    return stimuli / normalizing_factor.unsqueeze(-1)
