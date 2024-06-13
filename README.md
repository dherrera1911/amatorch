# ama_library

The package `ama_library` is a PyTorch implementation of the model
"Accuracy Maximization Analysis (AMA)", that learns optimal local
linear filters for a given task. The AMA model has been used to study
different visual tasks, such as estimation of 2D speed, binocular disparity,
defocus and 3D stereo motion.

AMA involves a learning linear filters whose outputs are decoded
probabilistically to perform a task. A description of the model can be found
in the paper "Accuracy Maximization Analysis for Sensory-Perceptual Tasks:
Computational Improvements, Filter Robustness, and Coding Advantages for
Scaled Additive Noise" J. Burge; P. Jaini. PLoS Computational Biology (2017)
and in the tutorials of this repository.

Originally the model was implemented in MATLAB
(https://github.com/burgelab/AMA). The present PyTorch implementation
provides much faster training and an object oriented interface.
The design of the package also allows for flexibility in
creating new versions of the model.

## Installation

The package can be installed using pip. First clone the repository and then
run the following command in the root directory of the repository:

```bash
pip install -e .
```

The package requires that the user manually installs the
package `geotorch`
(https://github.com/lezcano/geotorch[https://github.com/lezcano/geotorch]).


## Usage

The package provides a class `AMA_emp` that can be used to train and test
the AMA model. Detailed explanations of the model and its usage can be found
in the tutorials of this repository.


## Development

The package is still under development. A final version will be released
together with a publication.


