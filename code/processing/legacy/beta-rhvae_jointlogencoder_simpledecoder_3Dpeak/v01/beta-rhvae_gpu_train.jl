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

# Import GPU support
import CUDA

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
n_epoch = 100
# Define how often to save model
n_error = 1
# Define number of samples in batch
n_batch = 256
# Define fraction to split data into training and validation
split_frac = 0.85
# Define learning rate
η = 10^-4

# Define loss function hyper-parameters
ϵ = Float32(1E-4) # Leapfrog step size
K = 5 # Number of leapfrog steps
βₒ = 0.3f0 # Initial temperature for tempering

# Define temperatures to test
T_vals = [0.8f0, 0.6f0, 0.4f0, 0.2f0]

# Define ELBO prefactors
logp_prefactor = [10.0f0, 1.0f0, 1.0f0]
logq_prefactor = [1.0f0, 1.0f0, 1.0f0]

# Define RHVAE hyper-parameters in a dictionary
rhvae_kwargs = Dict(
    :K => K,
    :ϵ => ϵ,
    :βₒ => βₒ,
)

# Define loss function hyper-parameters in a dictionary
loss_kwargs = Dict(
    :K => K,
    :ϵ => ϵ,
    :βₒ => βₒ,
    :logp_prefactor => logp_prefactor,
    :logq_prefactor => logq_prefactor,
)
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

# Upload data to GPU
train_data = Flux.gpu(train_data)
val_data = Flux.gpu(val_data)

# Extract batches
train_batches = Flux.gpu.([x for x in train_loader])

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
rhvae_template = JLD2.load("./output/model.jld2")["model"]
# Load parameters
model_state = JLD2.load("./output/model.jld2")["model_state"]
# Input parameters to model
Flux.loadmodel!(rhvae_template, model_state)

## =============================================================================

println("Writing down metadata to README.md file")

# Define text to go into README
readme = """
# `$(@__FILE__)`
## Latent space dimensionalities to train
`latent_dim = $(size(rhvae_template.vae.encoder.µ.weight, 1))`
## Number of epochs
`n_epoch = $(n_epoch)`
## Training regimen
`rhvae mini-batch training witm multiple temperatures on the GPU`
## T values
`T_vals = $(T_vals)`
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
for i in eachindex(T_vals)
    # Define λ value
    T = T_vals[i]
    # Make a copy of the model   
    rhvae = AutoEncode.RHVAEs.RHVAE(
        deepcopy(rhvae_template.vae),
        deepcopy(rhvae_template.metric_chain),
        deepcopy(rhvae_template.centroids_data),
        deepcopy(rhvae_template.centroids_latent),
        deepcopy(rhvae_template.L),
        deepcopy(rhvae_template.M),
        T,
        deepcopy(rhvae_template.λ)
    )

    # Update metric parameters
    AutoEncode.RHVAEs.update_metric!(rhvae)

    # List previous model parameters
    model_states = sort(
        Glob.glob("$(out_dir)/gpu_beta-rhvae_$(T)temp_*epoch.jld2")
    )

    # Check if model states exist
    if length(model_states) > 0
        # Load model state
        model_state = JLD2.load(model_states[end])["model_state"]
        # Input parameters to model
        Flux.loadmodel!(rhvae, model_state)
        # Update metric parameters
        AutoEncode.RHVAEs.update_metric!(rhvae)
        # Extract epoch number
        epoch_init = parse(
            Int, match(r"epoch(\d+)", model_states[end]).captures[1]
        )
    else
        epoch_init = 1
    end # if

    ## =========================================================================

    # Explicit setup of optimizer
    opt_rhvae = Flux.Train.setup(
        Flux.Optimisers.Adam(η),
        rhvae
    )

    ## =========================================================================

    println("Training RHVAE...\n")

    # Loop through number of epochs
    for epoch in epoch_init:n_epoch
        # Loop through batches
        for (i, x) in enumerate(train_batches)
            # Train RHVAE
            AutoEncode.RHVAEs.train!(
                rhvae, x, opt_rhvae; loss_kwargs=loss_kwargs
            )
        end # for train_loader

        # Compute loss in training data
        loss_train = AutoEncode.RHVAEs.loss(
            rhvae, train_data; loss_kwargs...
        )
        # Compute loss in validation data
        loss_val = AutoEncode.RHVAEs.loss(
            rhvae, val_data; loss_kwargs...
        )

        # Forward pass training data
        local rhvae_train = rhvae(train_data; rhvae_kwargs...)
        # Compute MSE for training data
        local mse_train = Flux.mse(rhvae_train.µ, train_data)

        # Forward pass validation data
        local rhvae_val = rhvae(val_data; rhvae_kwargs...)
        # Compute MSE for validation data
        local mse_val = Flux.mse(rhvae_val.µ, val_data)


        println("\n ($(T)) Epoch: $(epoch) / $(n_epoch)\n - mse_train: $(mse_train)\n - mse_val: $(mse_val)\n")

        # Check if epoch is multiple of n_error
        if epoch % n_error == 0
            println("($T) saving epoch $(epoch)...")
            # Save checkpoint
            JLD2.jldsave(
                "$(out_dir)/beta-rhvae_$(T)temp_$(lpad(epoch, 5, "0"))epoch.jld2",
                model_state=Flux.cpu(Flux.state(rhvae)),
                mse_train=mse_train,
                mse_val=mse_val,
                loss_train=loss_train,
                loss_val=loss_val
            )
        end # if
    end # for n_epoch
end # for ϵ_vals
