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

# Define dimensionality of latent space
n_latent = 2
# Define number of neurons in hidden layers
n_neuron = 128

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

## =============================================================================

println("Defining VAE architecture")

println("Define JointLogEncoder...")
# Define encoder chain
encoder_chain = Flux.Chain(
    # First layer
    Flux.Dense(n_env => n_neuron, Flux.identity),
    # Second layer
    Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
    # Third layer
    Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
    # Fourth layer
    Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
)

# Define layers for µ and log(σ)
µ_layer = Flux.Dense(n_neuron => n_latent, Flux.identity)
logσ_layer = Flux.Dense(n_neuron => n_latent, Flux.identity)

# build encoder
encoder = AutoEncode.JointLogEncoder(encoder_chain, µ_layer, logσ_layer)

## =============================================================================

println("Define SplitLogDecoder...")
# Initialize decoder mean
decoder_µ = Flux.Chain(
    # First layer
    Flux.Dense(n_latent => n_neuron, Flux.identity),
    # Second Layer
    Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
    # Third layer
    Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
    # Fourth layer
    Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
    # Output layer
    Flux.Dense(n_neuron => n_env, Flux.identity)
)

# Initialize decoder log(σ)
decoder_logσ = Flux.Chain(
    # First layer
    Flux.Dense(n_latent => n_neuron, Flux.identity),
    # Second Layer
    Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
    # Third layer
    Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
    # Fourth layer
    Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
    # Output layer
    Flux.Dense(n_neuron => n_env, Flux.identity)
)

# Build splitlogdecoder
decoder = AutoEncode.SplitLogDecoder(decoder_µ, decoder_logσ)

## =============================================================================

# Initialize vae
vae = encoder * decoder

## =============================================================================

println("Save model object...")

# Save model object
JLD2.save(
    "./output/model.jld2",
    Dict("model" => vae, "model_state" => Flux.state(vae))
)