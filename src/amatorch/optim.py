import time

import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

__all__ = ["fit"]


def __dir__():
    return __all__


def fit(
    model,
    stimuli,
    labels,
    epochs,
    loss_fun=None,
    batch_size=512,
    learning_rate=0.1,
    decay_step=1000,
    decay_rate=1,
):
    """
    Learn AMA filters using Gradient Descent.

    Parameters
    ----------
    model : AMA model object
        The model used for fitting.
    stimuli : torch.Tensor
        Stimuli tensor of shape (n_stim, n_channels, n_dim).
    labels : torch.Tensor
        Label tensor of shape (n_stim).
    epochs : int
        Number of training epochs.
    loss_fun : callable, optional
        Loss function that takes in model, stimuli, and labels.
        Default is negative log posterior at the true category (cross-entropy).
    batch_size : int, optional
        Batch size, by default 512.
    learning_rate : float, optional
        Initial learning rate, by default 0.1.
    decay_step : int, optional
        Number of steps to decay the learning rate, by default 1000.
    decay_rate : float, optional
        Learning rate decay factor, by default 1.

    Returns
    -------
    torch.Tensor
        Tensor containing the loss at each epoch (shape: epochs).
    torch.Tensor
        Tensor containing the training time at each epoch (shape: epochs).
    """
    # Create data loader
    dataset = TensorDataset(stimuli, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    n_batches = len(data_loader)

    if loss_fun is None:
        loss_fun = kl_loss

    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=decay_step, gamma=decay_rate
    )
    optimizer.zero_grad()

    loss = []
    training_time = []
    total_start_time = time.time()
    prev_loss = None

    for e in tqdm(range(epochs), desc="Epochs", unit="epoch"):
        epoch_start_time = time.time()
        running_loss = 0.0

        for batch_stimuli, batch_labels in tqdm(
            data_loader, desc=f"Epoch {e+1}/{epochs}", unit="batch", leave=False
        ):

            optimizer.zero_grad()
            batch_loss = loss_fun(model, batch_stimuli, batch_labels)
            batch_loss.backward()
            optimizer.step()
            running_loss += batch_loss.detach().item()

        scheduler.step()

        epoch_time = time.time() - epoch_start_time
        training_time.append(epoch_time)
        current_loss = running_loss / n_batches

        loss_change = 0.0 if prev_loss is None else current_loss - prev_loss

        loss.append(current_loss)
        prev_loss = current_loss
        total_time = time.time() - total_start_time

        # Update tqdm bar description with loss change and total time
        tqdm.write(
            f"Epoch {e+1}/{epochs}, Loss: {current_loss:.4f}, "
            + f"Change: {loss_change:.4f}, Time: {total_time:.2f}s"
        )

    return torch.as_tensor(loss), torch.as_tensor(training_time)


def kl_loss(model, stimuli, labels):
    """
    Compute the negative log-likelihood loss (KL loss) for the AMA model.

    Parameters
    ----------
    model : AMA model object
        The model used for loss computation.
    stimuli : torch.Tensor
        Input stimuli tensor of shape (batch_size, n_features).
    labels : torch.Tensor
        True category labels for the stimuli as a vector of category indices.

    Returns
    -------
    torch.Tensor
        Negative log-likelihood loss.
    """
    n_stimuli = stimuli.shape[0]
    log_posteriors = torch.log(model.posteriors(stimuli) + 1e-8)
    correct_log_posteriors = log_posteriors[torch.arange(n_stimuli), labels]
    loss = -torch.mean(correct_log_posteriors)
    return loss
