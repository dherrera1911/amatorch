import matplotlib.pyplot as plt
import pytest

import amatorch.plot
from amatorch.datasets import disparity_data, disparity_filters
from amatorch.models import AMAGauss


@pytest.fixture(scope="module")
def model_outputs():
    data = disparity_data()
    filters = disparity_filters()

    ama = AMAGauss(
      stimuli=data["stimuli"],
      labels=data["labels"],
      n_filters=2,
      c50=0.5
    )
    ama.filters = filters

    responses = ama.get_responses(data["stimuli"])
    posteriors = ama.get_posteriors(data["stimuli"])
    estimates_labels = ama.get_estimates(data["stimuli"])
    estimates_values = data["values"][estimates_labels]
    true_values = data["values"][data["labels"]]
    return {
        "responses": responses,
        "posteriors": posteriors,
        "estimates_labels": estimates_labels,
        "estimates_values": estimates_values,
        "true_values": true_values,
        "labels": data["labels"],
        "class_values": data["values"],
        "ama": ama
    }


def test_filters_plot(model_outputs):
    """Test that the scatter of model responses works."""
    fig = amatorch.plot.plot_filters(model=model_outputs["ama"], n_cols=2)

    n_filters = model_outputs["ama"].filters.shape[0]

    axes = fig.get_axes()
    assert len(axes) == n_filters, "There should be n_filters axes in the filters plot."


def test_response_plot(model_outputs):
    """Test that the scatter of model responses works."""
    N_POINTS = 10
    n_classes = len(model_outputs["class_values"])

    fig, ax = plt.subplots()
    responses = model_outputs["responses"]

    amatorch.plot.scatter_responses(
      responses=responses,
      labels=model_outputs["labels"],
      values=model_outputs["class_values"],
      ax=ax,
      n_points=N_POINTS
    )

    collection = ax.collections[0]
    offsets = collection.get_offsets()
    assert offsets.shape[0] == N_POINTS * n_classes
    assert offsets.shape[1] == 2

