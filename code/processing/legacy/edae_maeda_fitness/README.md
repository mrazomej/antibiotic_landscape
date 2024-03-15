## `scripts`

- `edae_singlemask_crossvalidation_batch.jl`: Script that trains `n_cross`
  *emphasis denoising autoencoders* (edae) using the `Flux.jl` library. The
  training takes a fraction `train_frac` of the data for training and the rest
  for cross-validation. Each individual training model is stored as
  ```
  ./output/$(n_epoch)_epoch/edae_singlemask_$(lpad(cross, 2, "0"))_$(lpad(latent_dim, 2, "0"))dimensions.bson
  ```
  where `cross` refers to the number of cross-validation training.
  This `BSON` file contains three objects:
    - `ae`: The trained autoencoder.
    - `mse_train`: The loss function evaluated on the training dataset.
    - `mse_test`: The loss function evaluated on the test dataset.
    - `data_train`: data used to train the network.
    - `data_test`: data used to cross-validate the network.