import torch

def unit_norm(stimuli, c50=0):
    """
    Divide each channel of the stimuli by its offsetted norm
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
    # Normalizing factors
    offset_norm = torch.sqrt(torch.sum(stimuli**2, dim=-1) + c50)
    # Normalize
    return stimuli / offset_norm.unsqueeze(-1)

