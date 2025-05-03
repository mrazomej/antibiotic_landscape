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
n_batch = 128
# Define learning rate
η = 10^-3
# Define fraction of data to be used for training
split_frac = 0.85

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
data_dir = "$(git_root())/output/barbay_kinsler_2020/" *
           "advi_meanfield_hierarchicalreplicate_inference"

# Load data
df_kinsler = CSV.read(
    "$(data_dir)/kinsler_hierarchical_hyperfitness.csv",
    DF.DataFrame
)

# Define number of environmenst
n_env = length(unique(df_kinsler.env))

# Pivot to extract standardized mean and standard deviation fitness values
df_kinsler_mean = DF.unstack(df_kinsler, :env, :id, :fitness_mean_standard)

# Extract fitness matrix
data_mean = Float32.(Matrix(df_kinsler_mean[:, DF.Not(:env)]))

# Split indexes of data into training and validation
train_idx, val_idx = Flux.splitobs(
    1:size(data_mean, 2), at=split_frac, shuffle=true
)

# Extract train and validation data
train_data_mean = data_mean[:, train_idx] |> Flux.gpu
val_data_mean = data_mean[:, val_idx] |> Flux.gpu

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
    num_batches = size(train_data_mean, 2) ÷ n_batch
    # Shuffle data indexes
    idx_shuffle = Random.shuffle(1:size(train_data_mean, 2))
    # Split indexes into batches
    idx_batches = IterTools.partition(idx_shuffle, n_batch)
    # Loop through batches
    for (i, idx_tuple) in enumerate(idx_batches)
        println("Epoch: $(epoch) | Batch: $(i) / $(length(idx_batches))")
        # Extract indexes
        idx_batch = collect(idx_tuple)
        # Define train data as mean + standard deviation * standard normal
        train_data = train_data_mean[:, idx_batch]
        # Train RHVAE
        loss_epoch = AET.RHVAEs.train!(
            rhvae, train_data, opt_rhvae;
            loss_kwargs=loss_kwargs, verbose=false, loss_return=true
        )
        println("Loss: $(loss_epoch)")
    end # for train_loader

    # Sample train data as mean + standard deviation * standard normal
    train_sample = train_data_mean
    # Sample val data as mean + standard deviation * standard normal
    val_sample = val_data_mean

    println("Computing loss in training and validation data...")
    loss_train = AET.RHVAEs.loss(rhvae, train_sample; loss_kwargs...)
    loss_val = AET.RHVAEs.loss(rhvae, val_sample; loss_kwargs...)

    # Forward pass sample through model
    println("Computing MSE in training and validation data...")
    out_train = rhvae(train_sample; rhvae_kwargs...).μ
    mse_train = Flux.mse(train_sample, out_train)
    out_val = rhvae(val_sample; rhvae_kwargs...).μ
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
        "$(out_dir)/beta-rhvae_epoch$(lpad(epoch, 5, "0")).jld2",
        model_state=Flux.state(rhvae) |> Flux.cpu,
        loss_train=loss_train,
        loss_val=loss_val,
        mse_train=mse_train,
        mse_val=mse_val,
        train_idx=train_idx,
        val_idx=val_idx,
    )
end # for n_epoch
