## =============================================================================
println("Loading packages...")

# Import project package
import AutoEncode

# Import libraries to handel data
import CSV
import DataFrames as DF

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
n_epoch = 5_000
# Define number of samples in batch
n_batch = 64
# Define learning rate
η = 10^-3

## =============================================================================

println("Loading data into memory...")

# Define data directory
data_dir = "$(git_root())/data/Maeda_2020"

# Load file into memory
df_res = CSV.read("$(data_dir)/maeda_resistance_tidy.csv", DF.DataFrame)

# Initialize array to save data
ic50_mat = Matrix{Float32}(
    undef, length(unique(df_res.stress)), length(unique(df_res.strain))
)
# Group data by strain
df_group = DF.groupby(df_res, :strain, sort=true)

# Extract unique stresses to make sure the matrix is built correctly
stress = sort(unique(df_res.stress))

# Loop through groups
for (i, data) in enumerate(df_group)
    # Sort data by stress
    DF.sort!(data, :stress)
    # Check that the stress are in the correct order
    if all(data.stress .== stress)
        # Add data to matrix
        ic50_mat[:, i] = Float32.(data.ic50)
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

println("Preparing DataLoader...")

train_loader = Flux.DataLoader(ic50_std, batchsize=n_samples, shuffle=true)

## =============================================================================

println("Load model...\n")

# Load model
vae = JLD2.load("./output/model.jld2")["model"]
# Load parameters
model_state = JLD2.load("./output/model.jld2")["model_state"]
# Input parameters to model
Flux.loadmodel!(vae, model_state)

## =============================================================================

# Explicit setup of optimizer
opt_vae = Flux.Train.setup(
    # Flux.Optimisers.AdaDelta(),
    Flux.Optimisers.Adam(η),
    # Flux.AMSGrad(),
    vae
)

## =============================================================================

println("Setting output directories...")

# Define output directory
out_dir = "./output/$(n_epoch)_epoch"

# Check if output directory exists
if !isdir("./output/")
    mkdir("./output/")
end # if

# Check if output directory exists
if !isdir(out_dir)
    mkdir(out_dir)
end # if

## =============================================================================

println("Training vae...\n")
# Loop through number of epochs
for epoch in 1:n_epoch
    # Loop through batches
    for (i, x) in enumerate(train_loader)
        # Check if epoch is multiple of 100
        if epoch % 100 == 0
            println("\nEpoch: $(epoch)\n")
            println("Batch: $(i) / $(length(train_loader))")
            # Train vae
            AutoEncode.VAEs.train!(vae, x, opt_vae; verbose=true)
            # Forward pass batch data
            vae_outputs = vae(x)
            # Print MSE
            println("MSE: $(Flux.mse(vae_outputs.µ, x))")
            # Save checkpoint
            JLD2.jldsave(
                "$(out_dir)/vae_epoch$(lpad(epoch, 5, "0")).jld2",
                model_state=Flux.state(vae)
            )
        else
            # Train vae
            AutoEncode.VAEs.train!(vae, x, opt_vae; verbose=false)
        end # if
    end # for train_loader
end # for n_epoch

# Save final model
JLD2.jldsave(
    "$(out_dir)/vae_epoch$(lpad(n_epoch, 5, "0"))_final.jld2",
    model_state=Flux.state(vae)
)