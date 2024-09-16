# `RHVAE v01`

## `RHVAE` specific features

- latent space dimensionality: 2
- number of neurons in hidden layers: 128
- activation function: leaky ReLU
- temperature parameter: 0.8
- regularization parameter: 1E-2
- number of centroids: 256
- centroid determination: k-medoids

## `RHVAE` training

- training data: logic50 mcmc samples
- loss function hyperparameters:
    - leapfrog step size: 1E-4
    - leapfrog steps: 5
    - initial tempering temperature: 0.3
    - ELBO prefactors:
        - logp_prefactor = [10.0f0, 1.0f0, 1.0f0]
        - logq_prefactor = [1.0f0, 1.0f0, 1.0f0]
- optimizer: ADAM
- learning rate: 1E-4
- samples in batch: 128
- samples to compute loss: 256
