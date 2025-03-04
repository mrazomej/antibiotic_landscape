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

# Allow scalar
# CUDA.allowscalar(true)

## =============================================================================

# Define model hyperparameters

# Define number of epochs
n_epoch = 15
# Define number of samples in batch
n_batch = 512
# Define number of samples when computing loss
n_batch_loss = 512
# Define learning rate
η = 10^-3
# Define fractions of data to be used for training
split_fracs = [0.85, 0.75, 0.50, 0.25]

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
state_dir = "$(vae_dir)/model_crossvalidation_state"

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
# Define number of training environments
n_env_train = n_env ÷ 2

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
fit_std_input = StatsBase.transform(dt, fit_mat)[1:n_env_train, :]
fit_std_output = StatsBase.transform(dt, fit_mat)[n_env_train+1:end, :]

## =============================================================================

# Loop over fractions of data to be used for training
for (i, split_frac) in enumerate(split_fracs)

    println("Training RHVAE with $(split_frac) of data for training...")

    ## =========================================================================

    # Define output directory
    split_dir = "$(state_dir)/split_$(lpad(Int(split_frac * 100), 2, "0"))train_" *
                "$(lpad(Int(100 - split_frac * 100), 2, "0"))val"

    # Generate output directory if it doesn't exist
    if !isdir(split_dir)
        println("Generating output directory for $(split_frac) of data for training...")
        mkpath(split_dir)
    end

    ## =========================================================================

    println("Checking previous model states...")

    # List previous model parameters
    model_states = sort(Glob.glob("$(split_dir)/beta-rhvae_epoch*.jld2"[2:end], "/"))

    # Check if model states exist
    if 0 < length(model_states) < n_epoch
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
    elseif length(model_states) ≥ n_epoch
        println("Model states already exist for $(n_epoch) epochs")
        continue
    else
        epoch_init = 1
    end # if

    println("Initial epoch: $epoch_init")

    ## =========================================================================

    println("Splitting data into training and validation...")

    # Split indexes of data into training and validation
    train_idx, val_idx = Flux.splitobs(
        1:size(fit_std_input, 2), at=split_frac, shuffle=true
    )

    # Extract train and validation data
    train_data_input = fit_std_input[:, train_idx] |> Flux.gpu
    val_data_input = fit_std_input[:, val_idx] |> Flux.gpu

    train_data_output = fit_std_output[:, train_idx] |> Flux.gpu
    val_data_output = fit_std_output[:, val_idx] |> Flux.gpu

    ## =============================================================================

    println("Load model...\n")

    # Load model
    rhvae = JLD2.load("$(vae_dir)/model.jld2")["model"]

    # List previous model parameters
    model_state_file = sort(
        Glob.glob("$(vae_dir)/model_state/beta-rhvae_epoch*.jld2"[2:end], "/")
    )[end]

    # Load model state
    model_state = JLD2.load(model_state_file)["model_state"]

    # Input parameters to cross-validation model except for the decoder
    Flux.loadmodel!(rhvae.metric_chain, model_state.metric_chain)
    Flux.loadmodel!(rhvae.vae.encoder, model_state.vae.encoder)

    # Update metric parameters
    AET.RHVAEs.update_metric!(rhvae)

    ## =========================================================================

    println("Uploading model to GPU...")

    # Upload model to GPU
    rhvae = Flux.gpu(rhvae)

    # Explicit setup of optimizer
    opt_rhvae = Flux.Train.setup(
        Flux.Optimisers.Adam(η),
        rhvae
    )

    # Freeze enconder and metric chain
    Flux.freeze!(opt_rhvae.vae.encoder)
    Flux.freeze!(opt_rhvae.metric_chain)

    ## =============================================================================

    println("\nTraining RHVAE...\n")

    # Loop through number of epochs
    for epoch in epoch_init:n_epoch
        # Define number of batches
        num_batches = size(train_data_input, 2) ÷ n_batch
        # Shuffle data indexes
        idx_shuffle = Random.shuffle(1:size(train_data_input, 2))
        # Split indexes into batches
        idx_batches = IterTools.partition(idx_shuffle, n_batch)
        # Loop through batches
        for (i, idx_tuple) in enumerate(idx_batches)
            println("Split $(split_frac) | Epoch: $(epoch) | " *
                    "Batch: $(i) / $(length(idx_batches))")
            # Extract indexes
            idx_batch = collect(idx_tuple)
            # Train RHVAE
            loss_epoch = AET.RHVAEs.train!(
                rhvae,
                train_data_input[:, idx_batch, 1],
                train_data_output[:, idx_batch, 1],
                opt_rhvae;
                loss_kwargs=loss_kwargs, verbose=false, loss_return=true
            )
            println("Loss: $(loss_epoch)")
        end # for train_loader

        # Sample train data
        train_sample_idx = StatsBase.sample(
            1:size(train_data_input, 2), n_batch_loss, replace=false
        )
        # Sample train data
        train_sample_input = train_data_input[:, train_sample_idx]
        train_sample_output = train_data_output[:, train_sample_idx]

        # Sample val data
        val_sample_idx = StatsBase.sample(
            1:size(val_data_input, 2), n_batch_loss, replace=false
        )
        # Sample val data
        val_sample_input = val_data_input[:, val_sample_idx]
        val_sample_output = val_data_output[:, val_sample_idx]

        println("Computing loss in training and validation data...")
        loss_train = AET.RHVAEs.loss(
            rhvae,
            train_sample_input,
            train_sample_output;
            loss_kwargs...
        )
        loss_val = AET.RHVAEs.loss(
            rhvae,
            val_sample_input,
            val_sample_output;
            loss_kwargs...
        )

        # Forward pass sample through model
        println("Computing MSE in training and validation data...")
        out_train = rhvae(train_sample_input; rhvae_kwargs...).μ
        mse_train = Flux.mse(train_sample_output, out_train)
        out_val = rhvae(val_sample_input; rhvae_kwargs...).μ
        mse_val = Flux.mse(val_sample_output, out_val)

        println(
            "\n Split $(split_frac) | Epoch: $(epoch) / $(n_epoch)\n " *
            "   - loss_train: $(loss_train)\n" *
            "   - loss_val: $(loss_val)\n" *
            "   - mse_train: $(mse_train)\n" *
            "   - mse_val: $(mse_val)\n"
        )

        # Save checkpoint
        JLD2.jldsave(
            "$(split_dir)/beta-rhvae_epoch$(lpad(epoch, 5, "0")).jld2",
            model_state=Flux.state(rhvae) |> Flux.cpu,
            loss_train=loss_train,
            loss_val=loss_val,
            mse_train=mse_train,
            mse_val=mse_val,
            train_idx=train_idx,
            val_idx=val_idx,
        )
    end # for n_epoch

end # for split_frac in split_fracs