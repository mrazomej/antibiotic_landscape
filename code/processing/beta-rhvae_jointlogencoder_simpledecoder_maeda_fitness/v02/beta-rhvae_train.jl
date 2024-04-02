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
n_epoch = 1_000
# Define number of samples in batch
n_batch = 64
# Define fraction to split data into training and validation
split_frac = 0.85
# Define learning rates to test
learning_rates = [10^-5, 10^-4, 10^-3.5, 10^-3]

# Define loss function hyper-parameters
K = 5 # Number of leapfrog steps
ϵ = 1.0f-4 # Leapfrog step size
βₒ = 0.3f0 # Initial temperature for tempering

# Define RHVAE hyper-parameters in a dictionary
rhvae_kwargs = Dict(
    :K => K,
    :ϵ => ϵ,
    :βₒ => βₒ,
    :∇H => AutoEncode.RHVAEs.∇hamiltonian_TaylorDiff,
)

# Define pre-factors for loss function
logp_prefactor = [10.0f0, 1.0f0, 1.0f0]
logq_prefactor = [1.0f0, 1.0f0, 1.0f0]

# Define loss function hyper-parameters as a copy of the RHVAE hyper-parameters
# with extra pre-factor
loss_kwargs = deepcopy(rhvae_kwargs)
loss_kwargs[:logp_prefactor] = logp_prefactor
loss_kwargs[:logq_prefactor] = logq_prefactor


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
rhvae_template = JLD2.load("./output/model.jld2")["model"]
# Load parameters
model_state = JLD2.load("./output/model.jld2")["model_state"]
# Input parameters to model
Flux.loadmodel!(rhvae_template, model_state)
# Update metric parameters
AutoEncode.RHVAEs.update_metric!(rhvae_template)

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
`β-rhvae mini-batch training witm multiple learning rates`
## Learning rates
`η = $(learning_rates)`
## Batch size
`n_batch = $(n_batch)`
## Data split fraction
`split_frac = $(split_frac)`
## Optimizer
`Adam()`
"""

# Write README file into memory
open("./output/model_state/README.md", "w") do file
    write(file, readme)
end

## ============================================================================= 

# Loop through learning rates values
Threads.@threads for i in eachindex(learning_rates)
    # Define model
    rhvae = deepcopy(rhvae_template)

    # Extract learning rate
    η = learning_rates[i]

    # List previous model parameters
    model_states = sort(Glob.glob("$(out_dir)/beta-rhvae_$(η)learning-rate_epoch*.jld2"))

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
        for (i, x) in enumerate(train_loader)
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

        println("\n ($(η)) Epoch: $(epoch) / $(n_epoch)\n - mse_train: $(mse_train)\n - mse_val: $(mse_val)\n")

        # Save checkpoint
        JLD2.jldsave(
            "$(out_dir)/beta-rhvae_$(η)learning-rate_epoch$(lpad(epoch, 5, "0")).jld2",
            model_state=Flux.state(rhvae),
            mse_train=mse_train,
            mse_val=mse_val,
            loss_train=loss_train,
            loss_val=loss_val
        )
    end # for n_epoch
end # for β_vals
