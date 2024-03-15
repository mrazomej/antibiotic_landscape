## `scripts`

- `daegpu_batch.jl`: Script that trains a single denoising autoencoder using the
  `Flux.jl` library. The output of this script is stored as
  ```
  ./output/$(n_epoch)_epoch/dae_$(latent_dim)dimensions.bson
  ```
  This `BSON` file contains three objects:
    - `ae`: The trained autoencoder.
    - `mse`: The loss function as a function of epoch.
    - `data`: The IC50 data on which the training is based.
    - `data_std`: The Z-value scored data used to train the network.
- `daegpu_crossvalidation_batch.jl`: Script that trains `n_cross` denoising
  autoencoders using the `Flux.jl` library. The training takes a fraction
  `train_frac` of the data for training and the rest for cross-validation. Each
  individual training model is stored as
  ```
  ./output/$(n_epoch)_epoch/dae_$(lpad(cross, 2, "0"))_$(lpad(latent_dim, 2, "0"))dimensions.bson
  ```
  where `cross` refers to the number of cross-validation training.
  This `BSON` file contains three objects:
    - `ae`: The trained autoencoder.
    - `mse_train`: The loss function evaluated on the training dataset.
    - `mse_test`: The loss function evaluated on the test dataset.
    - `data_train`: data used to train the network.
    - `data_test`: data used to cross-validate the network.
- `daegpu_singlemask_crossvalidation_batch.jl`: Script that trains `n_cross`
  denoising autoencoders using the `Flux.jl` library. The training takes a
  fraction `train_frac` of the data for training and the rest for
  cross-validation. The difference with `daegpu_crossvalidation_batch.jl` is
  that when masking the components from the prediction, *all* of the entries on
  each mini-batch get the same elements masked. Each individual training model
  is stored as
  ```
  ./output/$(n_epoch)_epoch/dae_singlemask_$(lpad(cross, 2, "0"))_$(lpad(latent_dim, 2, "0"))dimensions.bson
  ```
  where `cross` refers to the number of cross-validation training.
  This `BSON` file contains three objects:
    - `ae`: The trained autoencoder.
    - `mse_train`: The loss function evaluated on the training dataset.
    - `mse_test`: The loss function evaluated on the test dataset.
    - `data_train`: data used to train the network.
    - `data_test`: data used to cross-validate the network.