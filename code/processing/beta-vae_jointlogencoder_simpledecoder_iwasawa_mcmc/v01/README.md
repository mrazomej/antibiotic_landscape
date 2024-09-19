# `VAE v01`

## `VAE` specific features

- latent space dimensionality: 2
- number of neurons in hidden layers: 128
- activation function: leaky ReLU

## `VAE` training

- training data: logic50 mcmc samples
- training split: 85% training, 15% validation
- loss function hyperparameters:
    - beta = 0.1
- optimizer: ADAM
- learning rate: 1E-3
- samples in batch: 256
- samples to compute loss: 256
- random seed: 42

## Notes

This VAE matches the structure of the RHVAE used on the same data.