# Neural network for optimization of optical resonator

## Abstract

This program aims to optimize optical resonators using convolutional neural networks.
Using resonance modes calculated by the finite element method as training data, the program optimizes the 2x2 dimensional parameters that determine the characteristics of the optical resonator.

The specific details of the structure are not disclosed, but it can be used to optimize general optical resonators,

## Functions 

- nntorch.py
  This is the main program of the repository.
It includes classes of neural networks, train, test, and loss functions, as well as the body of the training program.
The layer class " SymX" is defined to reflect the symmetry of the system in the explanatory variables.
- eval.py
  This is the script to evaluate the loss and accuracy (std) of trained model by using test data
- optimization.py
  To get the optimized variables from trained model.
- merge_coordinate.py
  Just for preprocessing


## Environment

Pytorch: 2.0.1+cu117 is intended for use with Pytorch: 2.0.1+cu117, but will work fine in a CPU environment.

