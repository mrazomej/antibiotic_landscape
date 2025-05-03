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
n_epoch = 50
# Define number of samples in batch
n_batch = 2048
# Define number of samples when computing loss
n_batch_loss = 2048
# Define learning rate
η = 10^-3
# Define fraction of data to be used for training
split_frac = 0.85

# Define loss function hyper-parameters
β = 0.1f0

# Define by how much to subsample the time series
n_sub = 1

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
state_dir = "$(vae_dir)/vae_model_state"

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
# Define step size
t_step = DD.dims(fitnotype_profiles, :time)[2] - t_init
# Subsample time series
fitnotype_profiles = fitnotype_profiles[time=DD.At(t_init:n_sub*t_step:t_final)]

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
train_data = fit_std[:, train_idx] |> Flux.gpu
val_data = fit_std[:, val_idx] |> Flux.gpu

## =============================================================================

println("Load model...\n")

# Load model
vae = JLD2.load("$(vae_dir)/vae_model.jld2")["model"]
# Load parameters
model_state = JLD2.load("$(vae_dir)/vae_model.jld2")["model_state"]
# Input parameters to model
Flux.loadmodel!(vae, model_state)

## =============================================================================

println("Checking previous model states...")

# List previous model parameters
model_states = sort(Glob.glob("$(state_dir)/beta-vae_epoch*.jld2"[2:end], "/"))

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
    idx_batches = collect(
        Base.Iterators.partition(idx_shuffle, n_batch)
    )[1:end-1]
    # Loop through batches
    for (i, idx_tuple) in enumerate(idx_batches)
        println("Epoch: $(epoch) | Batch: $(i) / $(length(idx_batches))")
        # Extract indexes
        idx_batch = collect(idx_tuple)
        # Train RHVAE
        loss_epoch = AET.VAEs.train!(
            vae, train_data[:, idx_batch], opt_vae;
            loss_kwargs=(β=β,), verbose=false, loss_return=true
        )
        println("Loss: $(loss_epoch)")
    end # for train_loader

    # Sample train data
    train_sample = train_data[
        :,
        StatsBase.sample(1:size(train_data, 2), n_batch_loss, replace=false),
    ]

    # Sample val data
    val_sample = val_data[
        :,
        StatsBase.sample(1:size(val_data, 2), n_batch_loss, replace=false),
    ]

    println("Computing loss in training and validation data...")
    loss_train = AET.VAEs.loss(vae, train_sample; β=β)
    loss_val = AET.VAEs.loss(vae, val_sample; β=β)

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

    if epoch % 10 == 0
        # Save checkpoint
        JLD2.jldsave(
            "$(state_dir)/beta-vae_epoch$(lpad(epoch, 5, "0")).jld2",
            model_state=Flux.state(vae) |> Flux.cpu,
            loss_train=loss_train,
            loss_val=loss_val,
            mse_train=mse_train,
            mse_val=mse_val,
            train_idx=train_idx,
            val_idx=val_idx,
        )
    else
        # Save checkpoint
        JLD2.jldsave(
            "$(state_dir)/beta-vae_epoch$(lpad(epoch, 5, "0")).jld2",
            loss_train=loss_train,
            loss_val=loss_val,
            mse_train=mse_train,
            mse_val=mse_val,
        )
    end
end # for n_epoch