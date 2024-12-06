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

# Define number of inputs
n_input = 3
# Define number of synthetic data points
n_data = 1_000

# Define model hyperparameters

# Define number of epochs
n_epoch = 10_000
# Define how often to save model
n_error = 50
# Define number of samples in batch
n_batch = 256
# Define fraction to split data into training and validation
split_frac = 0.85
# Define learning rate
η = 10^-3.5

# Defie loss function arguments
β = 1.0f0
α = 10.0f0

## =============================================================================

println("Generating synthetic data...")

# Define function
f(x₁, x₂) = 10.0f0 * exp(-(x₁^2 + x₂^2))

# Defien radius
radius = 3

# Sample random radius
r_rand = radius .* sqrt.(Random.rand(n_data))

# Sample random angles
θ_rand = 2π .* Random.rand(n_data)

# Convert form polar to cartesian coordinates
x_rand = Float32.(r_rand .* cos.(θ_rand))
y_rand = Float32.(r_rand .* sin.(θ_rand))
# Feed numbers to function
z_rand = f.(x_rand, y_rand)

# Compile data into matrix
data = Matrix(hcat(x_rand, y_rand, z_rand)')

# Fit model to standardize data to mean zero and standard deviation 1 on each
# environment
dt = StatsBase.fit(StatsBase.ZScoreTransform, data, dims=2)

# Center data to have mean zero and standard deviation one
data_std = StatsBase.transform(dt, data);

## =============================================================================

println("Preparing Data...")

# Assuming `ic50_std` is your data
train_data, val_data = Flux.splitobs(data_std, at=split_frac, shuffle=true)

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
infomaxvae = JLD2.load("./output/model.jld2")["model"]
# Load parameters
model_state = JLD2.load("./output/model.jld2")["model_state"]
# Input parameters to model
Flux.loadmodel!(infomaxvae, model_state)

## =============================================================================

println("Writing down metadata to README.md file")

# Define text to go into README
readme = """
# `$(@__FILE__)`
## Latent space dimensionalities to train
`latent_dim = $(size(infomaxvae.vae.encoder.µ.weight, 1))`
## Number of epochs
`n_epoch = $(n_epoch)`
## Training regimen
`InfoMaxVAE mini-batch training`
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

# List previous model parameters
model_states = sort(Glob.glob("$(out_dir)/infomaxvae_*epoch.jld2"))

# Check if model states exist
if length(model_states) > 0
    # Load model state
    model_state = JLD2.load(model_states[end])["model_state"]
    # Input parameters to model
    Flux.loadmodel!(infomaxvae, model_state)
    # Extract epoch number
    epoch_init = parse(
        Int, match(r"epoch(\d+)", model_states[end]).captures[1]
    )
else
    epoch_init = 1
end # if

## =========================================================================

# Explicit setup of optimizer
opt_infomaxvae = Flux.Train.setup(
    Flux.Optimisers.Adam(η),
    infomaxvae
)

## =========================================================================

println("Training infomaxvae...\n")

# Loop through number of epochs
for epoch in epoch_init:n_epoch
    # Loop through batches
    for (i, x) in enumerate(train_loader)
        # Train infomaxvae
        AutoEncode.InfoMaxVAEs.train!(infomaxvae, x, opt_infomaxvae)
    end # for train_loader

    # Check if epoch is multiple of n_error
    if epoch % n_error == 0

        println("Epoch: $(epoch)")
        # Compute loss in training data
        loss_train = AutoEncode.InfoMaxVAEs.infomaxloss(
            infomaxvae.vae, infomaxvae.mi, train_data;
            β=β,
            α=α
        )
        # Compute loss in validation data
        loss_val = AutoEncode.InfoMaxVAEs.infomaxloss(
            infomaxvae.vae, infomaxvae.mi, val_data;
            β=β,
            α=α
        )

        # Forward pass training data
        local infomaxvae_train = infomaxvae(train_data)
        # Compute MSE for training data
        local mse_train = Flux.mse(infomaxvae_train.µ, train_data)

        # Forward pass validation data
        local infomaxvae_val = infomaxvae(val_data)
        # Compute MSE for validation data
        local mse_val = Flux.mse(infomaxvae_val.µ, val_data)


        println("\n Epoch: $(epoch) / $(n_epoch)\n - mse_train: $(mse_train)\n - mse_val: $(mse_val)\n")
        println("saving epoch $(epoch)...")
        # Save checkpoint
        JLD2.jldsave(
            "$(out_dir)/infomaxvae_$(lpad(epoch, 5, "0"))epoch.jld2",
            model_state=Flux.state(infomaxvae),
            mse_train=mse_train,
            mse_val=mse_val,
            loss_train=loss_train,
            loss_val=loss_val
        )
    end # if
end # for n_epoch
