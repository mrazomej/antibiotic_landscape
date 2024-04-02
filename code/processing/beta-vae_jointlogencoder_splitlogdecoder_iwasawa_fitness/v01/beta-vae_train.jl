## =============================================================================
println("Loading packages...")

# Import project package
import AutoEncode

# Import libraries to handel data
import CSV
import DataFrames as DF
import Glob

# Import ML libraries
import Flux

# Import library to save models
import JLD2

# Import basic math
import StatsBase
import Random

# Import library to compute nearest neighbors
import NearestNeighbors

Random.seed!(42)

## =============================================================================

# Define model hyperparameters

# Define number of epochs
n_epoch = 20_000
# Define number of samples in batch
n_batch = 256
# Define fraction to split data into training and validation
split_frac = 0.85
# Define learning rate
η = 10^-5.5

# Define number of primary samples
n_primary = 16
# Define number of secondary samples
n_secondary = 16
# Define number of neighbors to consider
k_neighbors = 64

# Define β values to train
β_vals = Float32.([1E-3, 1E-2, 0.1, 0.25, 0.5, 0.75, 1.0])

## =============================================================================

println("Loading data into memory...")

# Define data directory
data_dir = "$(git_root())/data/Iwasawa_2022"

# Load file into memory
df_ic50 = CSV.read("$(data_dir)/iwasawa_ic50_tidy.csv", DF.DataFrame)

# Locate strains with missing values
missing_strains = unique(df_ic50[ismissing.(df_ic50.log2ic50), :strain])

# Remove data
df_ic50 = df_ic50[[x ∉ missing_strains for x in df_ic50.strain], :]

# Group data by strain and day
df_group = DF.groupby(df_ic50, [:strain, :day])

# Extract unique drugs to make sure the matrix is built correctly
drug = sort(unique(df_ic50.drug))

# Initialize matrix to save ic50 values
ic50_mat = Matrix{Float32}(undef, length(drug), length(df_group))

# Loop through groups
for (i, data) in enumerate(df_group)
    # Sort data by stress
    DF.sort!(data, :drug)
    # Check that the stress are in the correct order
    if all(data.drug .== drug)
        # Add data to matrix
        ic50_mat[:, i] = Float32.(data.log2ic50)
    else
        println("group $i stress does not match")
    end # if
end # for

# Define number of environments
n_env = size(ic50_mat, 1)
# Define number of samples
n_samples = size(ic50_mat, 2)

# Fit model to standardize data to mean zero and standard deviation 1 on each
# environment
dt = StatsBase.fit(StatsBase.ZScoreTransform, ic50_mat, dims=2)

# Center data to have mean zero and standard deviation one
ic50_std = StatsBase.transform(dt, ic50_mat)

## =============================================================================

println("Preparing Data...")

# Assuming `ic50_std` is your data
train_data, val_data = Flux.splitobs(ic50_std, at=split_frac, shuffle=true)

# Compute nearest neighbor tree
nn_tree = NearestNeighbors.BruteTree(train_data)

## =============================================================================

println("Setting output directories...")

# Define output directory
out_dir = "./output/model_state/"

# Check if output directory exists
if !isdir("./output/")
    mkdir("./output/")
end # if

# Check if output directory exists
if !isdir(out_dir)
    mkdir(out_dir)
end # if

## ============================================================================= 

println("Load model...\n")

# Load model
vae_template = JLD2.load("./output/model.jld2")["model"]
# Load parameters
model_state = JLD2.load("./output/model.jld2")["model_state"]
# Input parameters to model
Flux.loadmodel!(vae_template, model_state)

# Make a copy of the model for each β value
models = [Flux.deepcopy(vae_template) for _ in β_vals]

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
```
β-vae mini-batch training witm multiple β values alternating between training
the mean and training the log standard deviation. Mini-batches are selected with
the nearest neighbor tree.
```
## β values
`β_vals = $(β_vals)
## Batch size
`n_batch = $(n_batch)`
## Data split fraction
`split_frac = $(split_frac)`
## Optimizer
`Adam($η)`
"""

# Write README file into memory
open("./output/model_state/README.md", "w") do file
    write(file, readme)
end

## ============================================================================= 

# Loop through β values
Threads.@threads for i in eachindex(β_vals)
    # Define model
    vae = models[i]
    # Define β value
    β = β_vals[i]

    # List previous model parameters
    model_states = sort(Glob.glob("$(out_dir)/beta-vae_$(β)beta_epoch*.jld2"))

    # Check if model states exist
    if length(model_states) > 0
        # Load model state
        model_state = JLD2.load(model_states[end])["model_state"]
        # Input parameters to model
        Flux.loadmodel!(vae, model_state)
        # Extract epoch number
        epoch_init = parse(
            Int, match(r"epoch(\d+)", model_states[end]).captures[1]
        )
    else
        epoch_init = 1
    end # if

    ## =========================================================================

    # Explicit setup of optimizer
    opt_vae = Flux.Train.setup(
        Flux.Optimisers.Adam(η),
        vae
    )

    ## =========================================================================

    println("Training vae...\n")

    # Loop through number of epochs
    for epoch in epoch_init:n_epoch
        # Select batch
        data_batch = AutoEncode.utils.locality_sampler(
            train_data, nn_tree, n_primary, n_secondary, k_neighbors
        )

        # 1) Train VAE mean
        # Freeze decoder log(σ)
        Flux.freeze!(opt_vae.decoder.logσ)
        # Train vae
        AutoEncode.VAEs.train!(
            vae, data_batch, opt_vae; loss_kwargs=Dict(:β => β)
        )
        # Thaw decoder log(σ)
        Flux.thaw!(opt_vae.decoder.logσ)

        # 2) Train VAE log(σ)
        # Freeze decoder mean
        Flux.freeze!(opt_vae.decoder.µ)
        # Train vae
        AutoEncode.VAEs.train!(
            vae, data_batch, opt_vae; loss_kwargs=Dict(:β => β)
        )
        # Thaw decoder mean
        Flux.thaw!(opt_vae.decoder.µ)

        # Check if checkpoint should be saved
        if epoch % 250 == 0
            println("\n ($(β)) Epoch: $(epoch) / $(n_epoch)\n")

            # Compute loss in training data
            loss_train = AutoEncode.VAEs.loss(vae, train_data; β=β)
            # Compute loss in validation data
            loss_val = AutoEncode.VAEs.loss(vae, val_data; β=β)

            # Forward pass training data
            local vae_train = vae(train_data)
            # Compute MSE for training data
            local mse_train = Flux.mse(vae_train.µ, train_data)

            # Forward pass validation data
            local vae_val = vae(val_data)
            # Compute MSE for validation data
            local mse_val = Flux.mse(vae_val.µ, val_data)

            # Save checkpoint
            JLD2.jldsave(
                "$(out_dir)/beta-vae_$(β)beta_epoch$(lpad(epoch, 5, "0")).jld2",
                model_state=Flux.state(vae),
                mse_train=mse_train,
                mse_val=mse_val,
                loss_train=loss_train,
                loss_val=loss_val
            )
        end # if
    end # for n_epoch
end # for β_vals