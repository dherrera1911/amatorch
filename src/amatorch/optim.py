import numpy as np
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import time

# Define loop function to train the model
def fit(model, stimuli, labels, epochs, loss_fun, batch_size=512,
        learning_rate=0.1, decay_step=1000, decay_rate=1):
    """
    Learn AMA filters

    ----------------
    Arguments:
    ----------------
      - model: AMA model object.
      - stimuli: Stimulus tensor (n_stim x n_channels x n_dim)
      - labels: Label tensor (n_stim)
      - epochs: Number of epochs
      - loss_fun: Loss function. Takes in model, stimuli, and labels.
      - batch_size: Batch size. Default is 512
      - learning_rate: Learning rate. Default is 0.1
      - decay_step: Number of steps to decay learning rate. Default is 1000
      - decay_rate: Rate of learning rate decay. Default is 1
    ----------------
    Outputs:
    ----------------
      - loss: Tensor with loss at each epoch (epochs)
      - training_time: Time at each epoch (epochs)
    """
    # Create data loader
    dataset = TensorDataset(stimuli, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    n_batches = len(data_loader)

    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
      optimizer, step_size=decay_step, gamma=decay_rate
    )
    optimizer.zero_grad()

    loss = []
    training_time = []
    for e in range(epochs):

        running_loss = 0.0
        for batch_stimuli, batch_labels in data_loader:
            optimizer.zero_grad()
            batch_loss = loss_fun(model, batch_stimuli, batch_labels)
            batch_loss.backward()
            optimizer.step()
            running_loss += batch_loss.detach().item()
        scheduler.step()
        loss.append(running_loss/n_batches)
    return torch.as_tensor(loss), torch.as_tensor(training_time)
