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
n_epoch = 50
# Define number of samples in batch
n_batch = 256
# Define number of samples when computing loss
n_batch_loss = 256
# Define learning rate
η = 10^-3
# Define fraction of data to be used for training
split_frac = 0.5

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
# Define pre-trained model state directory
state_dir = "$(git_root())/output$(prefix)/rhvae_model_state"
# Define output directory
out_dir = "$(git_root())/output$(prefix)/rhvae_crossvalidation_state"

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
# Load drug list
drug_list = JLD2.load(
    "$(data_dir)/logic50_preprocess.jld2"
)["drugs"]

# Split indexes of data into training and validation
train_idx, val_idx = Flux.splitobs(
    1:size(data, 2), at=split_frac, shuffle=true
)

## =============================================================================

# Loop over drugs
for (i, drug) in enumerate(drug_list)
    println("Training RHVAE for $(drug) removal...")

    # Define all indexes except the index of the drug to be removed
    env_list = sort(setdiff(1:size(data, 1), i))

    # Extract train and validation data (different inputs and outputs)
    train_input = data[env_list, train_idx, :] |> Flux.gpu
    train_output = data[i:i, train_idx, :] |> Flux.gpu
    val_input = data[env_list, val_idx, :] |> Flux.gpu
    val_output = data[i:i, val_idx, :] |> Flux.gpu

    # --------------------------------------------------------------------------

    println("Load model for $(drug) removal...")

    # Load model
    rhvae_full = JLD2.load("$(model_dir)/model_$(drug)rm.jld2")["model"]
    # List parameters
    param_files = sort(Glob.glob("$(state_dir)/beta-rhvae_$(drug)rm_epoch*.jld2"[2:end], "/"))
    # Load latest parameters
    model_state = JLD2.load(param_files[end])["model_state"]
    # Input parameters to model
    Flux.loadmodel!(rhvae_full, model_state)
    # Update metric parameters
    AET.RHVAEs.update_metric!(rhvae_full)
    # Load second decoder
    decoder_missing = JLD2.load("$(model_dir)/decoder_missing.jld2")["model"]
    # Build second rhvae
    rhvae = AET.RHVAEs.RHVAE(
        deepcopy(rhvae_full.vae.encoder) * decoder_missing,
        deepcopy(rhvae_full.metric_chain),
        deepcopy(rhvae_full.centroids_data),
        deepcopy(rhvae_full.T),
        deepcopy(rhvae_full.λ)
    )
    # Update metric parameters
    AET.RHVAEs.update_metric!(rhvae)

    # --------------------------------------------------------------------------

    println("Checking previous model states...")

    # List previous model parameters
    model_states = sort(Glob.glob("$(out_dir)/beta-rhvae_$(drug)rm_crossval_$(split_frac)split_epoch*.jld2"[2:end], "/"))

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

    println("Initial epoch for $(drug) removal: $epoch_init")

    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------

    println("\nTraining RHVAE for $(drug) removal...\n")

    # Loop through number of epochs
    for epoch in epoch_init:n_epoch
        # Define number of batches
        num_batches = size(train_input, 2) ÷ n_batch
        # Shuffle data indexes
        idx_shuffle = Random.shuffle(1:size(train_input, 2))
        # Split indexes into batches
        idx_batches = IterTools.partition(idx_shuffle, n_batch)
        # Loop through batches
        for (i, idx_tuple) in enumerate(idx_batches)
            println("Epoch: $(epoch) | Batch: $(i) / $(length(idx_batches))")
            # Extract indexes
            idx_batch = collect(idx_tuple)
            # Sample mcmc index for the MCMC sample for this training step
            idx_mcmc = StatsBase.sample(1:size(train_input, 3))
            # Train RHVAE
            loss_epoch = AET.RHVAEs.train!(
                rhvae,
                train_input[:, idx_batch, idx_mcmc],
                train_output[:, idx_batch, idx_mcmc],
                opt_rhvae;
                loss_kwargs=loss_kwargs, verbose=false, loss_return=true
            )
            println("Loss: $(loss_epoch)")

            # Clear intermediate variables explicitly
            if i % 10 == 0  # Every 10 batches
                CUDA.reclaim(true)  # Force reclaim
                GC.gc(true)         # Force garbage collection

                # Optional: Add a small sleep to allow GPU cooling
                # sleep(0.1)
            end
        end # for train_loader

        # At end of epoch
        CUDA.reclaim()
        GC.gc()

        # Optional: Print GPU memory status
        println("GPU Memory: ", CUDA.memory_status())

        # Define train idx
        train_sample_idx = StatsBase.sample(
            1:size(train_input, 2), n_batch_loss, replace=false
        )
        train_mcmc_idx = StatsBase.sample(1:size(train_input, 3))
        # Sample train data
        train_sample_input = train_input[:, train_sample_idx, train_mcmc_idx]
        train_sample_output = train_output[:, train_sample_idx, train_mcmc_idx]
        # Define val idx
        val_sample_idx = StatsBase.sample(
            1:size(val_input, 2), n_batch_loss, replace=false
        )
        val_mcmc_idx = StatsBase.sample(1:size(val_input, 3))
        # Sample val data
        val_sample_input = val_input[:, val_sample_idx, val_mcmc_idx]
        val_sample_output = val_output[:, val_sample_idx, val_mcmc_idx]

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
            "\n Drug: $(drug) | $(i) / $(length(drug_list))\n " *
            "\n Epoch: $(epoch) / $(n_epoch)\n " *
            "   - loss_train: $(loss_train)\n" *
            "   - loss_val: $(loss_val)\n" *
            "   - mse_train: $(mse_train)\n" *
            "   - mse_val: $(mse_val)\n"
        )

        # Save checkpoint
        JLD2.jldsave(
            "$(out_dir)/beta-rhvae_$(drug)rm_crossval_$(split_frac)split_epoch$(lpad(epoch, 5, "0")).jld2",
            model_state=Flux.state(rhvae) |> Flux.cpu,
            loss_train=loss_train,
            loss_val=loss_val,
            mse_train=mse_train,
            mse_val=mse_val,
            train_idx=train_idx,
            val_idx=val_idx,
        )

    end # for n_epoch
end # for drug

