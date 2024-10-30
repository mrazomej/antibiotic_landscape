## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic

# Import project package
import AutoEncoderToolkit as AET

# Import packages for manipulating results
import DimensionalData as DD
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
n_epoch = 25
# Define number of samples in batch
n_batch = 512
# Define number of samples when computing loss
n_batch_loss = 512
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

# Define by how much to subsample the time series
n_sub = 10

## =============================================================================

println("Defining directories...")

# Locate current directory
path_dir = pwd()

# Find the path perfix where input data is stored
out_prefix = replace(
    match(r"processing/(.*)", path_dir).match,
    "processing" => "",
)

# Define output directory
out_dir = "$(git_root())/output$(out_prefix)"

# Generate output directory if it doesn't exist
if !isdir(out_dir)
    println("Generating output directory...")
    mkpath(out_dir)
end

# Define simulation directory
sim_dir = "$(out_dir)/sim_evo"

# Define VAE directory
vae_dir = "$(out_dir)/vae"

# Define model state directory
state_dir = "$(vae_dir)/model_state"

# Generate model state directory if it doesn't exist
if !isdir(state_dir)
    println("Generating model state directory...")
    mkpath(state_dir)
end

## =============================================================================

println("Loading data into memory...")

# Load fitnotype profiles
fitnotype_profiles = JLD2.load("$(sim_dir)/sim_evo.jld2")["fitnotype_profiles"]

# Extract initial and final time points
t_init, t_final = collect(DD.dims(fitnotype_profiles, :time)[[1, end]])
# Subsample time series
fitnotype_profiles = fitnotype_profiles[time=DD.At(t_init:n_sub:t_final)]

# Define number of environments
n_env = length(DD.dims(fitnotype_profiles, :landscape))

# Extract fitness data bringing the fitness dimension to the first dimension
fit_data = permutedims(fitnotype_profiles.fitness.data, (5, 1, 2, 3, 4, 6))
# Reshape the array to a Matrix
fit_data = reshape(fit_data, size(fit_data, 1), :)

# Reshape the array to stack the 3rd dimension
fit_mat = log.(fit_data)

# Fit model to standardize data to mean zero and standard deviation 1 on each
# environment 
dt = StatsBase.fit(StatsBase.ZScoreTransform, fit_mat, dims=2)

# Standardize the data to have mean 0 and standard deviation 1
fit_std = StatsBase.transform(dt, fit_mat)

# Split indexes of data into training and validation
train_idx, val_idx = Flux.splitobs(
    1:size(fit_std, 2), at=split_frac, shuffle=true
)

# Extract train and validation data
train_data = fit_std[:, train_idx, :] |> Flux.gpu
val_data = fit_std[:, val_idx, :] |> Flux.gpu

## =============================================================================

println("Load model...\n")

# Load model
rhvae = JLD2.load("$(vae_dir)/model.jld2")["model"]
# Load parameters
model_state = JLD2.load("$(vae_dir)/model.jld2")["model_state"]
# Input parameters to model
Flux.loadmodel!(rhvae, model_state)
# Update metric parameters
AET.RHVAEs.update_metric!(rhvae)

## =============================================================================

println("Checking previous model states...")

# List previous model parameters
model_states = sort(Glob.glob("$(state_dir)/beta-rhvae_epoch*.jld2"[2:end], "/"))

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
        # Train RHVAE
        loss_epoch = AET.RHVAEs.train!(
            rhvae, train_data[:, idx_batch, idx_mcmc], opt_rhvae;
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
        "$(state_dir)/beta-rhvae_epoch$(lpad(epoch, 5, "0")).jld2",
        model_state=Flux.state(rhvae) |> Flux.cpu,
        loss_train=loss_train,
        loss_val=loss_val,
        mse_train=mse_train,
        mse_val=mse_val,
        train_idx=train_idx,
        val_idx=val_idx,
    )
end # for n_epoch