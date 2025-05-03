# `RHVAE v05`

## `RHVAE` specific features

- latent space dimensionality: 2
- number of neurons in hidden layers: 128
- activation function: leaky ReLU
- temperature parameter: 0.5
- regularization parameter: 1E-2
- number of centroids: 256
- centroid determination: k-medoids

## `RHVAE` training

- training data: standardized posterior samples from the `BarBay.jl`
  **hierarchical and non-hierarchical replicate model** inference on the
  Kinsler et al. (2020) dataset. For datasets with multiple replicates, the
  hyper-fitness inferred from the hierarchical replicate model was used. For
  datasets with a single replicate, the non-hierarchical model was used.
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
- samples in batch: 256
- samples to compute loss: 256
- random seed: 42

## Notes

This training was performed using fitness posterior means obtained using
`BarBay.jl` hierarchical replicate model when available, otherwise the
non-hierarchical model was used.