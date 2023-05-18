## `scripts`

- `daegpu_batch.jl`: Script that trains a single denoising autoencoder using the
  `Flux.jl` library. The output of this script is stored as
  ```
  ./output/$(n_epoch)_epoch/ae_$(latent_dim)dimensions.bson
  ```
  This `BSON` file contains three objects:
    - `ae`: The trained autoencoder.
    - `mse`: The loss function as a function of epoch.
    - `data`: The IC50 data on which the training is based.
    - `data_std`: The Z-value scored data used to train the network.
