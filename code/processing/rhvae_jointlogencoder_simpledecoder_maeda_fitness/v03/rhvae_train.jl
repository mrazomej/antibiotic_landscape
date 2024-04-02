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
n_epoch = 2_500
# Define number of samples in batch
n_batch = 64
# Define learning rate
η = 1E-3

# Define loss function hyper-parameters
K = 5 # Number of leapfrog steps
ϵ = 1.0f-3 # Leapfrog step size
βₒ = 0.3f0 # Initial temperature for tempering

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

# Define loss function hyper-parameters as a copy of the RHVAE hyper-parameters
# with extra p_coeff parameter
loss_kwargs = deepcopy(rhvae_kwargs)

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
rhvae = JLD2.load("./output/model.jld2")["model"]
# Load parameters
model_state = JLD2.load("./output/model.jld2")["model_state"]
# Input parameters to model
Flux.loadmodel!(rhvae, model_state)
# Update metric parameters
AutoEncode.RHVAEs.update_metric!(rhvae)

# List previous model parameters
model_states = sort(Glob.glob("$(out_dir)/rhvae_epoch*.jld2"))

# Check if model states exist
if length(model_states) > 0
    # Load model state
    model_state = JLD2.load(model_states[end])["model_state"]
    # Input parameters to model
    Flux.loadmodel!(rhvae, model_state)
    # Update metric parameters
    AutoEncode.RHVAEs.update_metric!(rhvae)
    # Extract epoch number
    epoch_init = parse(Int, match(r"epoch(\d+)", model_states[end]).captures[1])
else
    epoch_init = 1
end # if

## =============================================================================

# Explicit setup of optimizer
opt_rhvae = Flux.Train.setup(
    Flux.Optimisers.Adam(η),
    rhvae
)

## =============================================================================

println("Training RHVAE...\n")

# Loop through number of epochs
for epoch in epoch_init:n_epoch
    println("\nEpoch: $(epoch)\n")
    # Loop through batches
    for (i, x) in enumerate(train_loader)
        println("Batch: $(i) / $(length(train_loader))")
        # Train RHVAE
        AutoEncode.RHVAEs.train!(
            rhvae, x, opt_rhvae; loss_kwargs=loss_kwargs, verbose=true
        )
        # Forward pass batch data
        rhvae_outputs = rhvae(x; rhvae_kwargs...)
        # Print MSE
        println("MSE: $(Flux.mse(rhvae_outputs.µ, x))")
    end # for train_loader
    # Save checkpoint
    JLD2.jldsave(
        "$(out_dir)/rhvae_epoch$(lpad(epoch, 5, "0")).jld2",
        model_state=Flux.state(rhvae)
    )
end # for n_epoch