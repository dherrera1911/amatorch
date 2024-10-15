import torch


def output_statistics(model_output, labels, quantiles=(0.025, 0.975)):
    """
    Compute the mean and sd of the output for each class.

    Parameters
    ----------
    model_output : torch.Tensor
        The output of the model (n_stimuli, n_classes) or (n_stimuli,).
    labels : torch.int64
        The true class for each stimulus (n_stimuli).

    Returns
    ---------
    output_dict : dict
        Dictionary containing statistics of the output for each class.

        mean : torch.Tensor
            The mean of the output for each class
            (n_classes, n_classes) or (n_classes,).
        sd : torch.Tensor
            The standard deviation of the output for each class
            (n_classes, n_classes) or (n_classes,).
        ci_low: torch.Tensor
            The lower bound of the 95% confidence interval of the output for
            each class (n_classes, n_classes) or (n_classes,).
        ci_high: torch.Tensor
            The upper bound of the 95% confidence interval of the output for
            each class (n_classes, n_classes) or (n_classes,).
        median: torch.Tensor
            The median of the output for each class
            (n_classes, n_classes) or (n_classes,).
    """
    n_classes = len(torch.unique(labels))
    output_dict = {"mean": [], "sd": [], "ci_low": [], "ci_high": [], "median": []}
    for i in range(n_classes):
        mask = labels == i
        output_dict["mean"].append(torch.mean(model_output[mask], dim=0))
        output_dict["sd"].append(torch.std(model_output[mask], dim=0))
        output_dict["ci_low"].append(
            torch.quantile(model_output[mask], quantiles[0], dim=0)
        )
        output_dict["ci_high"].append(
            torch.quantile(model_output[mask], quantiles[1], dim=0)
        )
        output_dict["median"].append(torch.median(model_output[mask], dim=0))
    return output_dict
