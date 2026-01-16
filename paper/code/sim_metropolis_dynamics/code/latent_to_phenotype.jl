## =============================================================================

println("Load packages")

# Import project package
import Antibiotic
import Antibiotic.mh as mh

# Import ML libraries
import AutoEncoderToolkit as AET
import Flux

# Import libraries to handle data
import Glob
import DimensionalData as DD
import JLD2

# Import basic math 
import Random
import StatsBase

# Set random seed
Random.seed!(42)

## =============================================================================

println("Defining directories...")

# Locate current directory
path_dir = pwd()

# Find the path perfix where input data is stored
out_prefix = replace(
    match(r"processing/(.*)", path_dir).match,
    "processing" => "",
)

# Define simulation directory
sim_dir = "$(git_root())/output$(out_prefix)/sim_evo"
# Define model directory
vae_dir = "$(git_root())/output$(out_prefix)/vae"
# Define output directory
state_dir = "$(vae_dir)/model_state"
# Define directory for neural network
mlp_dir = "$(git_root())/output$(out_prefix)/mlp"

# Generate neural network directory if it doesn't exist
if !isdir(mlp_dir)
    println("Generating neural network directory...")
    mkpath(mlp_dir)
end

## =============================================================================

println("Loading simulation results...")

# Define the subsampling interval
n_sub = 10

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
log_fitnotype_std = DD.DimArray(
    mapslices(slice -> StatsBase.transform(dt, slice),
        log.(fitnotype_profiles.fitness.data),
        dims=[5]),
    fitnotype_profiles.fitness.dims,
)

## =============================================================================

println("Loading model...")

# Find model file
model_file = first(Glob.glob("$(vae_dir)/model*.jld2"[2:end], "/"))
# List epoch parameters
model_states = sort(Glob.glob("$(state_dir)/*.jld2"[2:end], "/"))

# Load model
rhvae = JLD2.load(model_file)["model"]
# Load latest model state
Flux.loadmodel!(rhvae, JLD2.load(model_states[end])["model_state"])
# Update metric parameters
AET.RHVAEs.update_metric!(rhvae)

## =============================================================================

println("Mapping data to latent space...")

# Define latent space dimensions
latent = DD.Dim{:latent}([:latent1, :latent2])

# Map data to latent space
dd_latent = DD.DimArray(
    dropdims(
        mapslices(slice -> rhvae.vae.encoder(slice).μ,
            log_fitnotype_std.data,
            dims=[5]);
        dims=1
    ),
    (log_fitnotype_std.dims[2:4]..., latent, log_fitnotype_std.dims[6]),
)

# Reorder dimensions
dd_latent = permutedims(dd_latent, (4, 1, 2, 3, 5))

## =============================================================================

Random.seed!(42)

println("Defining neural network...")

# Define mlp output file
mlp_file = "$(mlp_dir)/latent_to_phenotype.jld2"

# Define neural network architecture
mlp_template = Flux.Chain(
    Flux.Dense(2 => 64, Flux.identity),
    Flux.Dense(64 => 64, Flux.leakyrelu),
    Flux.Dense(64 => 64, Flux.leakyrelu),
    Flux.Dense(64 => 64, Flux.leakyrelu),
    Flux.Dense(64 => 64, Flux.leakyrelu),
    Flux.Dense(64 => 2, Flux.identity),
)

# Move model to GPU (if available)
mlp_template = Flux.gpu(mlp_template)

# Save neural network
JLD2.save(mlp_file, Dict("mlp" => mlp_template))

## =============================================================================

"""
    prepare_training_data(dd_latent, fitnotype_profiles, split_frac)

Prepare training and validation data for a neural network that maps from latent space to phenotype space.

# Arguments
- `dd_latent`: DimensionalArray containing latent space coordinates
- `fitnotype_profiles`: DimensionalArray containing phenotype data
- `split_frac`: Fraction of data to use for training (between 0 and 1)

# Returns
Named tuple containing:
- `train`: Named tuple with standardized training data
    - `z`: GPU tensor of latent space coordinates
    - `x`: GPU tensor of phenotype values
- `val`: Named tuple with standardized validation data
    - `z`: GPU tensor of latent space coordinates 
    - `x`: GPU tensor of phenotype values
- `transforms`: Named tuple with standardization transforms
    - `z`: ZScoreTransform for latent space coordinates
    - `x`: ZScoreTransform for phenotype values
"""
function prepare_training_data(dd_latent, fitnotype_profiles, split_frac)
    # Format input data
    z_in = reduce(hcat, eachslice(dd_latent.data, dims=(2, 3, 4, 5), drop=true))
    # Format output data
    x_out = reduce(
        hcat,
        eachslice(
            fitnotype_profiles.phenotype[landscape=DD.At(1)].data,
            dims=(2, 3, 4, 5),
            drop=true
        )
    )

    # Standardize input and output data
    dz = StatsBase.fit(StatsBase.ZScoreTransform, z_in, dims=2)
    dx = StatsBase.fit(StatsBase.ZScoreTransform, x_out, dims=2)

    z_in_std = StatsBase.transform(dz, z_in)
    x_out_std = StatsBase.transform(dx, x_out)

    # Split data into training and validation sets
    train_idx, val_idx = Flux.splitobs(
        1:size(z_in, 2), at=split_frac, shuffle=true
    )

    # Create GPU tensors for training and validation
    z_in_train = z_in_std[:, train_idx] |> Flux.gpu
    z_in_val = z_in_std[:, val_idx] |> Flux.gpu
    x_out_train = x_out_std[:, train_idx] |> Flux.gpu
    x_out_val = x_out_std[:, val_idx] |> Flux.gpu

    return (
        train=(z=z_in_train, x=x_out_train),
        val=(z=z_in_val, x=x_out_val),
        transforms=(z=dz, x=dx)
    )
end

## =============================================================================

Random.seed!(42)

println("Training neural network...")

# Define fractions of data to use for training and validation
split_fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Define learning rate
η = 1E-4
# Define number of epochs
n_epochs = 1_000

# Loop through fractions of data to use for training and validation
Threads.@threads for i in 1:length(split_fracs)
    println("Training neural network for split $(split_fracs[i])...")
    # Define fraction of data to use for training and validation
    split_frac = split_fracs[i]
    # Define output file
    mlp_file_split = "$(mlp_dir)/latent_to_phenotype_split$(split_frac).jld2"
    # Prepare training and validation data
    data = prepare_training_data(dd_latent, fitnotype_profiles, split_frac)
    # Make copy of mlp
    mlp = deepcopy(mlp_template)
    # Explicit setup of optimizer
    opt_mlp = Flux.Train.setup(
        Flux.Optimisers.Adam(η),
        mlp
    )
    # Initialize loss vectors
    loss_train = []
    loss_val = []
    # Loop through number of epochs
    for epoch in 1:n_epochs
        # Compute loss and gradients
        loss, grads = Flux.withgradient(mlp) do mlp
            # Forward pass on input data
            x̂ = mlp(data.train.z)
            # Compute loss
            Flux.Losses.mse(x̂, data.train.x)
        end # end withgradient
        # Update model parameters
        Flux.update!(opt_mlp, mlp, grads[1])
        # Store loss
        push!(loss_train, loss)
        # Compute validation loss
        x̂_val = mlp(data.val.z)
        push!(loss_val, Flux.Losses.mse(x̂_val, data.val.x))
        # Print progress
        if epoch % 250 == 0
            println("Epoch $(epoch) of $(n_epochs) | Split $(split_frac)")
            println("   - Training loss: $(loss_train[end])")
            println("   - Validation loss: $(loss_val[end])")
        end # end if
    end # end loop through epochs
    # Save model state and loss vectors
    JLD2.save(
        mlp_file_split,
        Dict(
            "mlp_state" => Flux.state(mlp),
            "loss_train" => loss_train,
            "loss_val" => loss_val,
            "data" => data,
        )
    )
end # end loop through split fractions
