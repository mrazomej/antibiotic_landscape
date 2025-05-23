# metropolis-hastings evolution simulation

## simple description

- 2D phenotype space with 50 random evolution conditions.
    - small covariance making some peaks not reachable.
    - small inverse temperature making evolution less directed.

- 8-peak genetic density landscape.

## simulation setup

- phenotype space dimensionality: 2
- number of lineages: 5
- number of replicates: 3
- number of time steps: 300
- number of evolution conditions: 1
- type of evolution condition: random
- number of fitness landscapes: 50
- number of evolution conditions: 50
- type of mutational landscape: fixed
- mutational fitness landscape:
    - peak means:
        - [-1.5, -1.5]
        - [1.5, -1.5]
        - [1.5, 1.5]
        - [-1.5, 1.5]
    - peak amplitudes: 1.0
    - peak covariance: 0.45

## `RHVAE` specific features

- latent space dimensionality: 2
- number of neurons in hidden layers: 128
- activation function: leaky ReLU
- temperature parameter: 0.8
- regularization parameter: 1E-2
- number of centroids: 256
- centroid determination: k-medoids

## `RHVAE` training

- training data: standardized log fitness data
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

## files

- `sim_evo.jl`: main simulation script
- `sim_evo_viz.jl`: diagnostic plots for evolution simulation
- `rhvae_model.jl`: implementation of the RHVAE model
- `beta-rhvae_train.jl`: training script for the beta-RHVAE model
- `beta-rhvae_viz.jl`: diagnostic plots for the trained beta-RHVAE model
- `geodesic_model.jl`: implementation of the geodesic model
- `geodesic_train.jl`: training script for the geodesic model
- `geodesic_viz.jl`: diagnostic plots for the geodesic model
