import torch

def unit_norm(stimuli, c50=0):
    """Normalize stimuli to have norm less or equal to 1.

    Each channel is normalized by its offsetted norm
    (i.e. || sum of squares + c50 ||) and by the square
    root of the number of channels.
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

