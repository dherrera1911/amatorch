import matplotlib.pyplot as plt
import numpy as np


def plot_filters(model, n_cols=2, n_filters=10):
    """
    Plot the filters of an AMA model.

    Parameters
    ----------
    model : AMA model
        The model to plot the filters for.
    n_cols : int, optional
        Number of columns in the grid layout. Default is 3.
    n_filters : int, optional
        Number of filters to plot. Default is 10.
    """
    total_filters = model.filters.shape[0]
    n_filters = min(n_filters, total_filters)
    filters = model.filters[:n_filters].detach().cpu().numpy()

    n_channels, _ = filters.shape[1], filters.shape[2]
    n_rows = (n_filters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3), sharey=True
    )
    axes = np.array(axes).reshape(-1)

    # Color code for each channel
    colors = plt.cm.get_cmap("tab10", n_channels)

    for idx, ax in enumerate(axes[:n_filters]):
        filter_data = filters[idx]

        # Plot each channel of the filter
        for ch in range(n_channels):
            ax.plot(filter_data[ch], color=colors(ch), label=f"Channel {ch+1}")

        ax.set_title(f"Filter {idx+1}", fontsize=14)
        # ax.set_xticks([])
        ax.set_ylabel("Weight")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center")

    # Hide unused axes
    for ax in axes[n_filters:]:
        ax.axis("off")

    return fig
