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
n_epoch = 102
# Define number of samples in batch
n_batch = 256
# Define number of samples when computing loss
n_batch_loss = 256
# Define learning rate
η = 10^-3

# Define loss function hyper-parameters
ϵ = Float32(1E-3) # Leapfrog step size
K = 10 # Number of leapfrog steps
βₒ = 0.3f0 # Initial temperature for tempering

# Define ELBO prefactors
logp_prefactor = [10.0f0, 0.1f0, 0.1f0]
logq_prefactor = [0.1f0, 0.1f0, 0.1f0]

# Define RHVAE hyper-parameters in a NamedTuple
rhvae_kwargs = (
    K=K,
    ϵ=ϵ,
    βₒ=βₒ,
)

# Define loss function kwargs in a NamedTuple
loss_kwargs = (
    K=K,
    ϵ=ϵ,
    βₒ=βₒ,
    logp_prefactor=logp_prefactor,
    logq_prefactor=logq_prefactor,
)

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
)["logic50_mcmc_std"] |> Flux.gpu

## =============================================================================

println("Load model...\n")

# Load model
rhvae = JLD2.load("$(model_dir)/model.jld2")["model"]
# Load parameters
model_state = JLD2.load("$(model_dir)/model.jld2")["model_state"]
# Input parameters to model
Flux.loadmodel!(rhvae, model_state)
# Update metric parameters
AET.RHVAEs.update_metric!(rhvae)

## =============================================================================

println("Writing down metadata to README.md file")

# Define text to go into README
readme = """
# `$(@__FILE__)`
## Latent space dimensionalities to train
`latent_dim = $(size(rhvae.vae.encoder.µ.weight, 1))`
## Number of epochs
`n_epoch = $(n_epoch)`
## Training regimen
`beta-rhvae mini-batch training on GPU`
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
model_states = sort(Glob.glob("$(out_dir)/beta-rhvae_epoch*.jld2"[2:end], "/"))

# Check if model states exist
if length(model_states) > 0
    # Load model state
    model_state = JLD2.load(model_states[end])["model_state"]
    # Input parameters to model
    Flux.loadmodel!(rhvae, model_state)
    # Update metric parameters
    AET.RHVAEs.update_metric!(rhvae)
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
rhvae = Flux.gpu(rhvae)

# Explicit setup of optimizer
opt_rhvae = Flux.Train.setup(
    Flux.Optimisers.Adam(η),
    rhvae
)

## =============================================================================

println("\nTraining RHVAE...\n")

# Loop through number of epochs
for epoch in epoch_init:n_epoch
    # Define number of batches
    num_batches = size(data, 2) ÷ n_batch
    # Shuffle data indexes
    idx_shuffle = Random.shuffle(1:size(data, 2))
    # Split indexes into batches
    idx_batches = IterTools.partition(idx_shuffle, n_batch)
    # Loop through batches
    for (i, idx_tuple) in enumerate(idx_batches)
        println("Epoch: $(epoch) | Batch: $(i) / $(length(idx_batches))")
        # Extract indexes
        idx_batch = collect(idx_tuple)
        # Sample mcmc index for the MCMC sample for this training step
        idx_mcmc = StatsBase.sample(1:size(data, 3))
        # Train RHVAE
        loss_epoch = AET.RHVAEs.train!(
            rhvae, data[:, idx_batch, idx_mcmc], opt_rhvae;
            loss_kwargs=loss_kwargs, verbose=false, loss_return=true
        )
        println("Loss: $(loss_epoch)")
    end # for train_loader

    # Sample train data
    train_sample = data[
        :,
        StatsBase.sample(1:size(data, 2), n_batch_loss, replace=false),
        StatsBase.sample(1:size(data, 3))
    ]

    println("Computing loss in training data...")
    loss_train = AET.RHVAEs.loss(rhvae, train_sample; loss_kwargs...)

    # Forward pass sample through model
    println("Computing MSE in training data...")
    out_train = rhvae(train_sample; rhvae_kwargs...).μ
    mse_train = Flux.mse(train_sample, out_train)

    println(
        "\n Epoch: $(epoch) / $(n_epoch)\n " *
        "   - loss_train: $(loss_train)\n" *
        "   - mse_train: $(mse_train)\n"
    )

    # Save checkpoint
    JLD2.jldsave(
        "$(out_dir)/beta-rhvae_epoch$(lpad(epoch, 5, "0")).jld2",
        model_state=Flux.state(rhvae) |> Flux.cpu,
        loss_train=loss_train,
        mse_train=mse_train,
    )
end # for n_epoch
