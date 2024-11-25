# `RHVAE v05`

## `RHVAE` specific features

- latent space dimensionality: 3
- number of neurons in hidden layers: 128
- activation function: leaky ReLU
- temperature parameter: 0.5
- regularization parameter: 1E-2
- number of centroids: 256
- centroid determination: k-medoids

## `RHVAE` training

- training data: logic50 mcmc samples
- training split: 85% training, 15% validation
- loss function hyperparameters:
    - leapfrog step size: 1E-3
    - leapfrog steps: 10
    - initial tempering temperature: 0.3
    - ELBO prefactors:
        - logp_prefactor = [10.0f0, 0.1f0, 0.1f0]
        - logq_prefactor = [0.1f0, 0.1f0, 0.1f0]
- optimizer: ADAM
- learning rate: 1E-3
- samples in batch: 64
- samples to compute loss: 256
- random seed: 42

## Notes

- This is a 3D latent space model.

- The number of samples in a batch was reduced to 64. This was the only way to
  get the loss to decrease, as larger batch sizes would result in a loss that
  would not decrease.

- This training regime includes a larger leapfrog step (10x compared to v04) and
  twice as many steps (from 5 to 10).

