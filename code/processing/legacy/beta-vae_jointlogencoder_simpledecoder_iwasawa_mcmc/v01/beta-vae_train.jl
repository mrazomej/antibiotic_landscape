## =============================================================================
println("Loading packages...")

# Import project package
import AutoEncoderToolkit as AET

# Import libraries to handel data
import CSV
import DataFrames as DF
import Glob

# Import ML libraries
import Flux

# Import CUDA
import CUDA

# Import library to save models
import JLD2

# Import itertools
import IterTools

# Import basic math
import StatsBase
import Random
Random.seed!(42)

## =============================================================================

# Define model hyperparameters

# Define number of epochs
n_epoch = 1_000
# Define number of samples in batch
n_batch = 256
# Define number of samples when computing loss
n_batch_loss = 256
# Define learning rate
η = 10^-3
# Define fraction of data to be used for training
split_frac = 0.85

# Define ELBO prefactors
β = 0.1f0

# Define loss kwargs named tuple
loss_kwargs = (β=β,)

## =============================================================================

println("Setting output directories...")

# Define current directory
path_dir = pwd()

# Find the path perfix where input data is stored
prefix = replace(
    match(r"processing/(.*)", path_dir).match,
    "processing" => "",
)

# Define model directory
model_dir = "$(git_root())/output$(prefix)"
# Define output directory
out_dir = "$(git_root())/output$(prefix)/model_state"

# Generate output directory if it doesn't exist
if !isdir(out_dir)
    println("Generating output directory...")
    mkpath(out_dir)
end

## =============================================================================

println("Loading data into memory...")

# Define data directory
data_dir = "$(git_root())/output/mcmc_iwasawa_logistic"

# Load standardized mean data
data = JLD2.load(
    "$(data_dir)/logic50_preprocess.jld2"
)["logic50_mcmc_std"]

# Split indexes of data into training and validation
train_idx, val_idx = Flux.splitobs(
    1:size(data, 2), at=split_frac, shuffle=true
)

# Extract train and validation data
train_data = data[:, train_idx, :] |> Flux.gpu
val_data = data[:, val_idx, :] |> Flux.gpu

## =============================================================================

println("Load model...\n")

# Load model
vae = JLD2.load("$(model_dir)/model.jld2")["model"]
# Load parameters
model_state = JLD2.load("$(model_dir)/model.jld2")["model_state"]
# Input parameters to model
Flux.loadmodel!(vae, model_state)

## =============================================================================

println("Writing down metadata to README.md file")

# Define text to go into README
readme = """
# `$(@__FILE__)`
## Latent space dimensionalities to train
`latent_dim = $(size(vae.encoder.µ.weight, 1))`
## Number of epochs
`n_epoch = $(n_epoch)`
## Training regimen
`beta-vae mini-batch training on GPU`
## Batch size
`n_batch = $(n_batch)`
## Optimizer
`Adam($η)`
"""

# Write README file into memory
open("$(out_dir)/README.md", "w") do file
    write(file, readme)
end

## =============================================================================

println("Checking previous model states...")

# List previous model parameters
model_states = sort(Glob.glob("$(out_dir)/beta-vae_epoch*.jld2"[2:end], "/"))

# Check if model states exist
if length(model_states) > 0
    # Load model state
    model_state = JLD2.load(model_states[end])["model_state"]
    # Input parameters to model
    Flux.loadmodel!(vae, model_state)
    # Extract epoch number
    epoch_init = parse(
        Int, match(r"epoch(\d+)", model_states[end]).captures[1]
    ) + 1
else
    epoch_init = 1
end # if

println("Initial epoch: $epoch_init")

## =============================================================================

println("Uploading model to GPU...")

# Upload model to GPU
vae = Flux.gpu(vae)

# Explicit setup of optimizer
opt_vae = Flux.Train.setup(
    Flux.Optimisers.Adam(η),
    vae
)

## =============================================================================

println("\nTraining VAE...\n")

# Loop through number of epochs
for epoch in epoch_init:n_epoch
    # Define number of batches
    num_batches = size(train_data, 2) ÷ n_batch
    # Shuffle data indexes
    idx_shuffle = Random.shuffle(1:size(train_data, 2))
    # Split indexes into batches
    idx_batches = IterTools.partition(idx_shuffle, n_batch)
    # Loop through batches
    for (i, idx_tuple) in enumerate(idx_batches)
        println("Epoch: $(epoch) | Batch: $(i) / $(length(idx_batches))")
        # Extract indexes
        idx_batch = collect(idx_tuple)
        # Sample mcmc index for the MCMC sample for this training step
        idx_mcmc = StatsBase.sample(1:size(train_data, 3))
        # Extract batch data
        train_batch = train_data[:, idx_batch, idx_mcmc]
        # Train VAE
        loss_epoch = AET.VAEs.train!(
            vae, train_batch, opt_vae;
            loss_kwargs=loss_kwargs, verbose=false, loss_return=true
        )
        println("Loss: $(loss_epoch)")
    end # for train_loader

    # Sample train data
    train_sample = train_data[
        :,
        StatsBase.sample(1:size(train_data, 2), n_batch_loss, replace=false),
        StatsBase.sample(1:size(train_data, 3))
    ]
    # Sample val data
    val_sample = val_data[:, :, StatsBase.sample(1:size(val_data, 3))]

    println("Computing loss in training and validation data...")
    loss_train = AET.VAEs.loss(vae, train_sample; loss_kwargs...)
    loss_val = AET.VAEs.loss(vae, val_sample; loss_kwargs...)

    # Forward pass sample through model
    println("Computing MSE in training and validation data...")
    out_train = vae(train_sample).μ
    mse_train = Flux.mse(train_sample, out_train)
    out_val = vae(val_sample).μ
    mse_val = Flux.mse(val_sample, out_val)

    println(
        "\n Epoch: $(epoch) / $(n_epoch)\n " *
        "   - loss_train: $(loss_train)\n" *
        "   - loss_val: $(loss_val)\n" *
        "   - mse_train: $(mse_train)\n" *
        "   - mse_val: $(mse_val)\n"
    )

    # Save checkpoint
    JLD2.jldsave(
        "$(out_dir)/beta-vae_epoch$(lpad(epoch, 5, "0")).jld2",
        model_state=Flux.state(vae) |> Flux.cpu,
        loss_train=loss_train,
        loss_val=loss_val,
        mse_train=mse_train,
        mse_val=mse_val,
        train_idx=train_idx,
        val_idx=val_idx,
    )
end # for n_epoch
