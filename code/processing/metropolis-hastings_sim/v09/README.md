# metropolis-hastings evolution simulation

## simple description

- 3D phenotype space with one fixed and 49 random evolution conditions.
    - large covariances to promote that from any position in phenotype
    space, there are multiple fitness peaks that can be reached.
    - large inverse temperature making evolution more directed.

- 3D latent space with 256 centroids determined by k-medoids.

- 85%/15% training/validation split.

## simulation setup

- phenotype space dimensionality: 3
- number of lineages: 10
- number of replicates: 2
- number of time steps: 300
- inverse temperature β: 100
- number of evolution conditions: 1
- type of evolution condition: random
- number of fitness landscapes: 50
- number of evolution conditions: 1 fixed, 49 random
- type of mutational landscape: fixed
- fixed evolution condition parameters:
    - peak mean: [0.0, 0.0, 0.0]
    - peak amplitude: 5.0
    - peak covariance: 3.0
- random evolution condition parameters:
    - peak amplitude range: [1.0, 5.0]
    - peak covariance range: [3.0, 10.0]
    - number of peaks range: [1, 4]
- fixed mutational fitness landscape:
    - peak means:
        - [-1.5, -1.5, -1.5]
        - [1.5, -1.5, -1.5]
        - [1.5, 1.5, -1.5]
        - [-1.5, 1.5, -1.5]
        - [-1.5, -1.5, 1.5]
        - [1.5, -1.5, 1.5]
        - [1.5, 1.5, 1.5]
        - [-1.5, 1.5, 1.5]
    - peak amplitudes: 1.0
    - peak covariance: 0.45

## `RHVAE` specific features

- latent space dimensionality: 3
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
- samples in batch: 512
- samples to compute loss: 512
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
