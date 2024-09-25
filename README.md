# Accuracy Maximization Analysis in Python

`ama_library` is a python library for learning optimal linear filters
to perform a given supervised task. The library uses the
"Accuracy Maximization Analysis (AMA)" method[^1]. AMA model has
been used to study different visual tasks, such as estimation of
2D speed, binocular disparity, defocus, and 3D stereo motion.

## Installation

The package requires Python 3.7 up to 3.10. If conda
is installed, create a new environment with the following command:

```bash
conda create -n my-ama python=3.10
```

Activate the environment:

```bash
conda activate my-ama
```

Then clone the repository and install using pip:

```bash
git clone git@github.com:dherrera1911/accuracy_maximization_analysis.git
cd accuracy_maximization_analysis
pip install -e .
```

## Usage

The package provides a class `AMA_emp` that can be used to train and test
the AMA model. Detailed explanations of the model and its usage can be found
in the tutorials of this repository.

## Development

This package is under development, and at a very early stage.


[^1] For a description of the model see
the paper "Accuracy Maximization Analysis for Sensory-Perceptual Tasks:
Computational Improvements, Filter Robustness, and Coding Advantages for
Scaled Additive Noise" J. Burge; P. Jaini. PLoS Computational Biology (2017),
and in the tutorials of this repository.
