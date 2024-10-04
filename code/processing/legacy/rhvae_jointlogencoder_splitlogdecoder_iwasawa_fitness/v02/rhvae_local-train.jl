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

# Import library to compute nearest neighbors
import NearestNeighbors

# Import basic math
import StatsBase
import Random
Random.seed!(42)

## =============================================================================

# Define model hyperparameters

# Define number of epochs
n_epoch = 2_500
# Define number of samples in batch
n_batch = 128
# Define learning rate
η = 1E-4

# Define number of primary samples
n_primary = 16
# Define number of secondary samples
n_secondary = 16
# Define number of neighbors to consider
k_neighbors = 64

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

println("Generating NearestNeighbors tree...")

# Compute nearest neighbor tree
nn_tree = NearestNeighbors.BruteTree(ic50_std)

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
    # Select batch
    x = AutoEncode.utils.locality_sampler(
        ic50_std, nn_tree, n_primary, n_secondary, k_neighbors
    )

    # 1) Train VAE mean
    # Freeze decoder log(σ)
    Flux.freeze!(opt_rhvae.vae.decoder.logσ)
    # Train RHVAE
    AutoEncode.RHVAEs.train!(
        rhvae, x, opt_rhvae; loss_kwargs=loss_kwargs, verbose=true
    )
    # Thaw decoder log(σ)
    Flux.thaw!(opt_rhvae.vae.decoder.logσ)

    # 2) Train VAE log(σ)
    # Freeze decoder mean
    Flux.freeze!(opt_rhvae.vae.decoder.µ)
    # Train RHVAE
    AutoEncode.RHVAEs.train!(
        rhvae, x, opt_rhvae; loss_kwargs=loss_kwargs, verbose=true
    )
    # Thaw decoder mean
    Flux.thaw!(opt_rhvae.vae.decoder.µ)

    # Forward pass batch data
    rhvae_outputs = rhvae(x; rhvae_kwargs...)

    # Print MSE
    println("MSE: $(Flux.mse(rhvae_outputs.µ, x))")
    # Save checkpoint
    JLD2.jldsave(
        "$(out_dir)/rhvae_epoch$(lpad(epoch, 5, "0")).jld2",
        model_state=Flux.state(rhvae)
    )
end # for n_epoch
