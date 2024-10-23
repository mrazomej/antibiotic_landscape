# metropolis-hastings evolution simulation

## simulation setup

- phenotype space dimensionality: 2
- number of lineages: 10
- number of time steps: 300
- number of evolution conditions: 1
- type of evolution condition: fixed
- evolution fitness landscape:
    - peak mean: [0, 0]
    - peak amplitude: 5.0
    - peak covariance: 3.0
- mutational fitness landscape:
    - peak means:
        - [-1.5, -1.5]
        - [1.5, -1.5]
        - [1.5, 1.5]
        - [-1.5, 1.5]
    - peak amplitudes: 1.0
    - peak covariance: 0.45
- number of non-evolution environments: 50
- type of non-evolution environment: random

## files

- `sim_evo.jl`: main simulation script
- `sim_evo_viz.jl`: diagnostic plots for evolution simulation
- `rhvae_model.jl`: implementation of the RHVAE model
- `beta-rhvae_train.jl`: training script for the beta-RHVAE model
- `beta-rhvae_viz.jl`: diagnostic plots for the trained beta-RHVAE model
- `geodesic_model.jl`: implementation of the geodesic model
- `geodesic_train.jl`: training script for the geodesic model
- `geodesic_viz.jl`: diagnostic plots for the geodesic model
