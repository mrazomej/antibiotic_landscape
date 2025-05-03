# `RHVAE rm trajectories v01`

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
- training split: All of the data from one evolution condition was used as the
  validation set, and the rest of the data was used as the training set.
- loss function hyperparameters:
    - leapfrog step size: 1E-3
    - leapfrog steps: 10
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

- This is the first test of removing an entire evolution condition from the
  training data.

- The centroids are computed from the all of the data, including the validation
  data.

- A model is trained for each evolution condition, where the validation data is
  the data from that evolution condition, and the training data is the data from
  all other evolution conditions.

- This training regime includes a large leapfrog step and twice as many steps
  (from 5 to 10).
