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

# Define number of hidden layers
n_hidden = 4
# Define number of neurons in non-linear hidden layers
n_neuron = 400
# Define dimensionality of latent space
n_latent = 2

# Define RHVAE hyper-parameters
T = 0.8f0 # Temperature
λ = 1.0f-2 # Regularization parameter
n_centroids = 32 # Number of centroids

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

# Select centroids via k-means
centroids_data = AutoEncode.utils.centroids_kmedoids(ic50_std, n_centroids)

## =============================================================================

println("Defining RHVAE architecture")

# Define latent space activation function
latent_activation = Flux.identity
# Define output layer activation function
output_activation = Flux.identity

# Define encoder layer and activation functions
encoder_neurons = repeat([n_neuron], n_hidden)
encoder_activation = [[Flux.identity]; repeat([Flux.relu], n_hidden - 1)]

# Define decoder layer and activation function
decoder_neurons = repeat([n_neuron], n_hidden)
decoder_activation = [[Flux.identity]; repeat([Flux.relu], n_hidden - 1)]

# Define MetricChain layer and activation function
metric_neurons = repeat([n_neuron], n_hidden)
metric_activation = [[Flux.identity]; repeat([Flux.relu], n_hidden - 1)]

## =============================================================================

println("Initializing rhvae...")

# Initialize encoder
encoder = AutoEncode.JointLogEncoder(
    n_env,
    n_latent,
    encoder_neurons,
    encoder_activation,
    latent_activation
)

# Initialize decoder
decoder = AutoEncode.SimpleDecoder(
    n_env,
    n_latent,
    decoder_neurons,
    decoder_activation,
    output_activation
)

# Define Metric MLP
metric_chain = AutoEncode.RHVAEs.MetricChain(
    n_env,
    n_latent,
    metric_neurons,
    metric_activation,
    Flux.identity
)


# Initialize rhvae
rhvae = AutoEncode.RHVAEs.RHVAE(
    encoder * decoder,
    metric_chain,
    centroids_data,
    T,
    λ
)

## =============================================================================

println("Save model object...")

# Save model object
JLD2.save(
    "./output/model.jld2",
    Dict("model" => rhvae, "model_state" => Flux.state(rhvae))
)