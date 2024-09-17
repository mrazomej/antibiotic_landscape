# `RHVAE v01`

## `RHVAE` specific features

- latent space dimensionality: 2
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
    - leapfrog step size: 1E-4
    - leapfrog steps: 5
    - initial tempering temperature: 0.3
    - ELBO prefactors:
        - logp_prefactor = [10.0f0, 0.1f0, 0.1f0]
        - logq_prefactor = [0.1f0, 0.1f0, 0.1f0]
- optimizer: ADAM
- learning rate: 1E-3
- samples in batch: 256
- samples to compute loss: 256
- random seed: 42

## Notes

This configuration and training parameters are aimed at reproducing the same
training regime as with the point estimates of the IC50 used before the last
update of `AutoEncoderToolkit.jl`
