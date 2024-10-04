##################

# TESTS THAT THE AMA CLASS INITIALIZES AND TRAINS
# FOR THE RELEASE BEFORE REFACTORING
#
##################

import pytest
import torch
import amatorch.ama_class as cl
from amatorch.data import disparity_data
import amatorch.optim as optim
import amatorch.utilities as au

# Initialize the AMA class
N_EPOCHS = 10
LR = 0.1
LR_STEP = 5
LR_GAMMA = 0.5
BATCH_SIZE = 512
RESPONSE_NOISE = 0.1
C50 = 0.5

######## TEST THAT AMA RUNS ########
def test_training():

    # Load the data
    data_dict = disparity_data()
    stimuli = data_dict['stimuli']
    labels = data_dict['labels']
    values = data_dict['values']

    ama = cl.AMAGauss(
      stimuli=stimuli,
      labels=labels,
      n_filters=2,
      response_noise=RESPONSE_NOISE,
      c50=C50,
    )

    # Fit model
    loss, training_time = optim.fit(
      model=ama, stimuli=stimuli, labels=labels,
      epochs=N_EPOCHS, loss_fun=au.kl_loss,
      batch_size=BATCH_SIZE, learning_rate=LR,
      decay_step=LR_STEP, decay_rate=LR_GAMMA,
    )

    # Get the posteriors
    posteriors = ama.posteriors(stimuli)

    # Sample from the distribution
    assert not torch.isnan(ama.filters.detach()).any(), 'Filters are nan'
    assert loss[0] > loss[-1], 'Loss did not decrease'
    assert not torch.isnan(posteriors).any(), 'Posteriors are nan'
