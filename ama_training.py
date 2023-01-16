import scipy.io as spio
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import geotorch
from ama_class import AMA
from ama_utilities import *


###### LOAD AMA DATA
# Load ama struct from .mat file into Python
ama = spio.loadmat('./data/ama_Z.mat')
# Extract contrast normalized, noisy stimulus
s = ama.get("s")
s = torch.from_numpy(s)
s = s.transpose(0,1)
# Extract the vector indicating category of each stimulus row
ctgInd = ama.get("ctgInd")
ctgInd = ctgInd.flatten()       # remove singleton dimension
ctgInd = ctgInd.astype(int)     # convert to int to use as index
ctgInd = ctgInd-1       # convert to python indexing (subtract 1)
ctgInd = np.ndarray.tolist(ctgInd)
ctgInd = torch.Tensor(ctgInd)
ctgInd = ctgInd.type(torch.LongTensor)  # convert to torch integer

# Extract the values of the latent variable
ctgVal = torch.arange(start=-2.5, end=2.51, step=0.125)

# Extract the filters learned with AMA in matlab, to compare to the Python ones
fOri = ama.get("f")
fOri = torch.from_numpy(fOri)
fOri = fOri.transpose(0,1)


##############
#### Set parameters of the model to train
##############
nFilt = 4 # Number of filters to use
filterSigma = 0.005 # Variance of filter responses 

# Choose training params
batch_size = 128*5
learning_rate = 0.005

##################
###### INITIALIZE AMA MODEL
##################

# Initialize model with random filters
amaPy = AMA(sAll=s, ctgInd=ctgInd, nFilt=nFilt, filterSigma=filterSigma)
# Add norm 1 constraint (set parameters f to lay on a sphere)
geotorch.sphere(amaPy, "f")

# Put data into Torch data loader tools
train_ds = TensorDataset(s, ctgInd)  # Stimuli and label to be loaded together
train_dl = DataLoader(train_ds, batch_size, shuffle=True)  # Batch loading and other utilities 

# Choose loss function
lossFun = nn.CrossEntropyLoss()

# Set up optimizer
opt = torch.optim.Adam(amaPy.parameters(), lr=learning_rate)  # Adam
#opt = torch.optim.SGD(amaPy.parameters(), lr=0.03)  # SGD


# Define loop function to train the model
def fit(num_epochs, model, loss_fn, opt):
    print('Training loss: ', loss_fn(model.get_posteriors(s), ctgInd))
    opt.zero_grad()
    for epoch in range(num_epochs):
        for sb, ctgb in train_dl:
            # Generate predictions for batch sb, returned by train_dl 
            model.update_response_statistics()
            pred = model.get_posteriors(sb)
            loss = loss_fn(pred, ctgb)  # Compute loss with associated categories of the batch
            # Perform gradient descent
            loss.backward()  # Compute gradient
            opt.step()       # Take one step
            opt.zero_grad()  # Delete gradient
        print('Training loss: ', loss_fn(model.get_posteriors(s), ctgInd))
    # Do the final response statistics update
    model.update_response_statistics()

fit(num_epochs=5, model=amaPy, loss_fn=lossFun, opt=opt)




# Visualize filters
fPy = amaPy.f.detach()  # Untrained Py filters
view_filters(fIn=fPy)
plt.show()


ppi = amaPy.get_posteriors(s)
est = amaPy.get_estimates(s)



