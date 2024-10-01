##################

# TESTS THAT THE AMA CLASS INITIALIZES AND TRAINS
# FOR THE RELEASE BEFORE REFACTORING
#
##################

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
import amatorch.ama_class as cl
import amatorch.optim as optim
from amatorch.data import disparity_data
import amatorch.utilities as au

# Initialize the AMA class
N_EPOCHS = 10
LR = 0.05
LR_STEP = 5
LR_GAMMA = 0.5
BATCH_SIZE = 512

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
      values=values,
    )

    # Train the model
    trainDataset = TensorDataset(stimuli, labels)
    # Batch loading and other utilities 
    trainDataLoader = DataLoader(
      trainDataset,
      batch_size=BATCH_SIZE,
      shuffle=True)
    # Optimizer and scheduler
    opt = torch.optim.Adam(ama.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.StepLR(
      opt, step_size=LR_STEP, gamma=LR_GAMMA
    )

    # Fit model
    loss, tstLoss, elapsedTimes = optim.fit(
      nEpochs=N_EPOCHS, model=ama, trainDataLoader=trainDataLoader,
      lossFun=au.kl_loss, opt=opt, scheduler=sch, sTst=stimuli,
      ctgIndTst=labels
    )

    # Get posteriors
    posteriors = ama.posteriors(stimuli)

    # Sample from the distribution
    assert not torch.isnan(ama.f.detach()).any(), 'Filters are nan'
    assert loss[0] > loss[-1], 'Loss did not decrease'
    assert not torch.isnan(posteriors).any(), 'Posteriors are nan'

