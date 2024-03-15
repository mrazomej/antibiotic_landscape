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

- `aesimple_crossvalidation_batch.jl`: Script that trains multiple autoencoders
  using the `SimpleChains.jl` library. This script trains on a fraction `p` on
  the data and evaluates the MSE on the `1 - p` fraction of data not used for
  training as a cross-validation scheme. The output of this script is stored as
  ```
  ./output/$(n_epoch)_epoch/ae_$(cross)_$(latent_dim)dimensions.bson
  ```
  where `cross` refers to the number of cross-validation training.
  This `BSON` file contains three objects:
    - `ae`: The trained autoencoder.
    - `mse_train`: The loss function evaluated on the training dataset.
    - `mse_test`: The loss function evaluated on the test dataset.
    - `data_train`: data used to train the network.
    - `data_test`: data used to cross-validate the network.