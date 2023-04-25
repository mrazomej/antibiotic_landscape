## `scripts`

- `aesimple_mean_batch.jl`: Script that trains a single autoencoder using the
  `SimpleChains.jl` library in a multi-thread fashion. This means that the
  gradient of the loss function is computed in multiple threads to speed-up the
  computation. The output of this script is stored as
  ```
  ./output/$(n_epoch)_epoch/ae_$(latent_dim)dimensions.bson
  ```
  This `BSON` file contains three objects:
    - `ae`: The trained autoencoder.
    - `mse_sc`: The loss function as computed by `SimpleChains.jl`.
    - `mse_flux`: The mean squared error as computed by mapping the
      `SimpleChains.jl` model to a `Flux.jl` model.
