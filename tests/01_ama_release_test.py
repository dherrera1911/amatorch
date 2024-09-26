##################
#
# TESTS THAT THE AMA CLASS INITIALIZES AND TRAINS
# FOR THE RELEASE BEFORE REFACTORING
#
##################

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
import ama_library.ama_class as cl
import ama_library.utilities as au
from utility_functions import load_disparity_data

# Initialize the AMA class
SAMPLES_PER_STIM = 5
RESP_NOISE_VAR = torch.tensor(0.003)
PIXEL_COV = torch.tensor(0.005)
N_EPOCHS = 10
LR = 0.05
LR_STEP = 5
LR_GAMMA = 0.5
BATCH_SIZE = 512

######## TEST THAT AMA RUNS ########
def test_ama_runs():

    # Load the data
    s, ctgInd, ctgVal = load_disparity_data()

    ama = cl.AMA_emp(
      sAll=s,
      ctgInd=ctgInd,
      nFilt=2,
      respNoiseVar=RESP_NOISE_VAR,
      pixelCov=PIXEL_COV,
      ctgVal=ctgVal,
      samplesPerStim=SAMPLES_PER_STIM,
      nChannels=2
    )

    # Train the model
    trainDataset = TensorDataset(s, ctgInd)
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
    loss, tstLoss, elapsedTimes = au.fit(
      nEpochs=N_EPOCHS, model=ama, trainDataLoader=trainDataLoader,
      lossFun=au.kl_loss, opt=opt, scheduler=sch, sTst=s,
      ctgIndTst=ctgInd
    )

    # Get posteriors
    posteriors = ama.get_posteriors(s)

    # Sample from the distribution
    assert not torch.isnan(ama.f.detach()).any(), 'Filters are nan'
    assert loss[0] > loss[-1], 'Loss did not decrease'

