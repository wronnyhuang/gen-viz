# Neural net vs linear model

This folder contains the code needed to train a neural network and linear model (on quadratic kernels) with similar number of parameters on CIFAR10.

## Requirements

This code requires the following packages
- Python 3.6
- Torch 1.0
- Torchvision 0.2.1

Different versions may likely work but they have not been tested

## Train the model

To train the neural network

`python main.py -gpu=0`

To train the linear model

`python main.py -gpu=0 -poly`
