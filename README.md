# AMAtorch

`amatorch` is a Python implementation of Accuracy Maximization Analysis (AMA),
a supervised dimensionality reduction method based on maximizing the accuracy
of a probabilistic decoder (i.e. ideal observer model) for a given task.
This method has been primarily used to obtain image-computable ideal
observers for a variety of visual tasks in the context of human vision.

## Overview

`amatorch` provides a set classes implementing variants of the AMA model,
and utility functions to train and test the model. AMA models
have two main components: an encoding stage with learnable filters,
and a decoding stage that uses filter response distributions
to estimate the stimulus class.

An AMA model can be initialized with a random set of filters and
trained as follows:

```python
import amatorch.optim as optim
from amatorch.datasets import disparity_data
from amatorch.models import AMAGauss

data = disparity_data()

ama = AMAGauss(
    stimuli=data["stimuli"],
    labels=data["labels"],
    n_filters=2,
    response_noise=0.05,
    c50=1.0,
)

# Fit model
loss, training_time = optim.fit(
    model=ama,
    stimuli=data["stimuli"],
    labels=data["labels"],
    epochs=20,
    batch_size=512,
    learning_rate=0.1,
    decay_step=4,
    decay_rate=0.5,
)
```

The resulting model can be used to obtain posterior probabilities
of the classes for a given stimulus, or the estimated class.

```python
posterior = ama.posteriors(data["stimuli"])
estimated_class = ama.estimates(data["stimuli"])
```

See the tutorials for more details on the model structure,
its variants and its usage.


## Installation

To install the package, clone the repository, go to the
downloaded directory and install using pip. In the command
line, this can be done as follows:

```bash
git clone git@github.com:dherrera1911/amatorch.git
cd amatorch
pip install -e .
```

For more detailed instructions, see the installation section
of the tutorials.


