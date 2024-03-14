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
Random.seed!(42)

## =============================================================================

# Define model hyperparameters

# Define number of epochs
n_epoch = 10_000
# Define number of samples in batch
n_batch = 256
# Define fraction to split data into training and validation
split_frac = 0.9
# Define learning rate
η = 10^-5.5

# Define number of cycles
n_cycles = 4
# Define β parameters
β_min = 0.0f0
β_max = 1.0f0

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

train_loader = Flux.DataLoader(train_data, batchsize=n_batch, shuffle=true)

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
vae = JLD2.load("./output/model.jld2")["model"]
# Load parameters
model_state = JLD2.load("./output/model.jld2")["model_state"]
# Input parameters to model
Flux.loadmodel!(vae, model_state)

# List previous model parameters
model_states = sort(Glob.glob("$(out_dir)/vae-cyc_epoch*.jld2"))

# Check if model states exist
if length(model_states) > 0
    # Load model state
    model_state = JLD2.load(model_states[end])["model_state"]
    # Input parameters to model
    Flux.loadmodel!(vae, model_state)
    # Extract epoch number
    epoch_init = parse(Int, match(r"epoch(\d+)", model_states[end]).captures[1])
else
    epoch_init = 1
end # if

## =============================================================================

# Explicit setup of optimizer
opt_vae = Flux.Train.setup(
    Flux.Optimisers.Adam(η),
    vae
)

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
`cyclical annealing mini-batch training [β_min = $(β_min), β_max = $(β_max)]`
## Number of annealing cycles
`n_cycles = $(n_cycles)`
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

println("Training vae...\n")

# Forward pass training data
vae_train = vae(train_data)
# Compute MSE for training data
mse_train = Flux.mse(vae_train.µ, train_data)

# Forward pass validation data
vae_val = vae(val_data)
# Compute MSE for validation data
mse_val = Flux.mse(vae_val.µ, val_data)

# Loop through number of epochs
for epoch in epoch_init:n_epoch
    # Define β parameter
    β = AutoEncode.utils.cycle_anneal(epoch, n_epoch, n_cycles)

    # Loop through batches
    for (i, x) in enumerate(train_loader)
        # Train vae
        AutoEncode.VAEs.train!(
            vae, x, opt_vae; verbose=false, loss_kwargs=Dict(:β => β)
        )
    end # for train_loader

    # Check if checkpoint should be saved
    if epoch % 150 == 0
        println("\nEpoch: $(epoch) / $(n_epoch)\n")

        println("β: $(β)\n")

        # Compute loss in training data
        loss_train = AutoEncode.VAEs.loss(vae, train_data)
        # Compute loss in validation data
        loss_val = AutoEncode.VAEs.loss(vae, val_data)

        # Print loss for training and validation data
        println("Loss Train: $(loss_train)")
        println("Loss Val: $(loss_val)\n")

        # Forward pass training data
        local vae_train = vae(train_data)
        # Compute MSE for training data
        local mse_train = Flux.mse(vae_train.µ, train_data)

        # Forward pass validation data
        local vae_val = vae(val_data)
        # Compute MSE for validation data
        local mse_val = Flux.mse(vae_val.µ, val_data)

        # Print MSE for training and validation data
        println("MSE Train: $(mse_train)")
        println("MSE Val: $(mse_val)\n")

        # Save checkpoint
        JLD2.jldsave(
            "$(out_dir)/vae_epoch$(lpad(epoch, 5, "0")).jld2",
            model_state=Flux.state(vae),
            mse_train=mse_train,
            mse_val=mse_val,
            loss_train=loss_train,
            loss_val=loss_val
        )
    end # if
end # for n_epoch
