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
n_batch = 256
# Define fraction to split data into training and validation
split_frac = 0.85
# Define learning rate
η = 10^-3

# Define loss function hyper-parameters
ϵ = Float32(1E-4) # Leapfrog step size
K = 5 # Number of leapfrog steps
βₒ = 0.3f0 # Initial temperature for tempering

# Define λ values
λ_vals = Float32.([1E-3, 1E-2, 1E-1, 1E0, 1E1])

# Define RHVAE hyper-parameters in a dictionary
rhvae_kwargs = Dict(
    :K => K,
    :ϵ => ϵ,
    :βₒ => βₒ,
    :∇H => AutoEncode.RHVAEs.∇hamiltonian_TaylorDiff,
    :∇H_kwargs => Dict(
        :momentum_logprior => AutoEncode.RHVAEs.riemannian_logprior_loop
    ),
)
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
rhvae_template = JLD2.load("./output/model.jld2")["model"]
# Load parameters
model_state = JLD2.load("./output/model.jld2")["model_state"]
# Input parameters to model
Flux.loadmodel!(rhvae_template, model_state)
# Update metric parameters
AutoEncode.RHVAEs.update_metric!(rhvae_template)

# Make a copy of the model for each β value
# models = [Flux.deepcopy(rhvae_template) for _ in λ_vals]

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
`rhvae mini-batch training witm multiple λ values`
## Lambda values
`λ_vals = $(λ_vals)`
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
Threads.@threads for i in eachindex(λ_vals)
    # Define λ value
    λ = λ_vals[i]
    # Make a copy of the model   
    rhvae = AutoEncode.RHVAEs.RHVAE(
        rhvae_template.vae,
        rhvae_template.metric_chain,
        rhvae_template.centroids_data,
        rhvae_template.centroids_latent,
        rhvae_template.L,
        rhvae_template.M,
        rhvae_template.T,
        λ
    )

    # Update metric parameters
    AutoEncode.RHVAEs.update_metric!(rhvae)

    # List previous model parameters
    model_states = sort(Glob.glob("$(out_dir)/rhvae_$(λ)lambda_epoch*.jld2"))

    # Check if model states exist
    if length(model_states) > 0
        # Load model state
        model_state = JLD2.load(model_states[end])["model_state"]
        # Input parameters to model
        Flux.loadmodel!(rhvae, model_state)
        # Update metric parameters
        Flux.update_metric!(rhvae)
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
                rhvae, x, opt_rhvae; loss_kwargs=rhvae_kwargs
            )
        end # for train_loader

        # Compute loss in training data
        loss_train = AutoEncode.RHVAEs.loss(
            rhvae, train_data; rhvae_kwargs...
        )
        # Compute loss in validation data
        loss_val = AutoEncode.RHVAEs.loss(
            rhvae, val_data; rhvae_kwargs...
        )

        # Forward pass training data
        local rhvae_train = rhvae(train_data; rhvae_kwargs...)
        # Compute MSE for training data
        local mse_train = Flux.mse(rhvae_train.µ, train_data)

        # Forward pass validation data
        local rhvae_val = rhvae(val_data; rhvae_kwargs...)
        # Compute MSE for validation data
        local mse_val = Flux.mse(rhvae_val.µ, val_data)


        println("\n ($(λ)) Epoch: $(epoch) / $(n_epoch)\n - mse_train: $(mse_train)\n - mse_val: $(mse_val)\n")

        # Save checkpoint
        JLD2.jldsave(
            "$(out_dir)/rhvae_$(λ)lambda_epoch$(lpad(epoch, 5, "0")).jld2",
            model_state=Flux.state(rhvae),
            mse_train=mse_train,
            mse_val=mse_val,
            loss_train=loss_train,
            loss_val=loss_val
        )
    end # for n_epoch
end # for λ_vals
