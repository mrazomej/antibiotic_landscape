## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic
import Antibiotic.mh as mh

# Import project package
import AutoEncoderToolkit

# Import packages for manipulating results
import DimensionalData as DD

# Import ML libraries
import Flux

# Import library to save models
import JLD2

# Import basic math
import StatsBase
import Random
Random.seed!(42)

## =============================================================================

println("Defining directories...")

# Locate current directory
path_dir = pwd()

# Find the path perfix where input data is stored
out_prefix = replace(
    match(r"processing/(.*)", path_dir).match,
    "processing" => "",
)

# Define output directory
out_dir = "$(git_root())/output$(out_prefix)"

# Generate output directory if it doesn't exist
if !isdir(out_dir)
    println("Generating output directory...")
    mkpath(out_dir)
end

# Define simulation directory
sim_dir = "$(out_dir)/sim_evo"

# Define VAE directory
vae_dir = "$(out_dir)/vae"

## =============================================================================

println("Defining VAE parameters...")

# Define dimensionality of latent space
n_latent = 2
# Define number of neurons in hidden layers
n_neuron = 128

# Define by how much to subsample the time series
n_sub = 10

## =============================================================================

println("Loading data into memory...")

# Load fitnotype profiles
fitnotype_profiles = JLD2.load("$(sim_dir)/sim_evo.jld2")["fitnotype_profiles"]

# Extract initial and final time points
t_init, t_final = collect(DD.dims(fitnotype_profiles, :time)[[1, end]])
# Subsample time series
fitnotype_profiles = fitnotype_profiles[time=DD.At(t_init:n_sub:t_final)]

# Define number of environments
n_env = length(DD.dims(fitnotype_profiles, :landscape))
# Define number of training environments
n_env_train = n_env ÷ 2

## =============================================================================

println("Defining VAE architecture")

println("Define JointGaussianLogEncoder...")
# Define encoder chain
encoder_chain = Flux.Chain(
    # First layer
    Flux.Dense(n_env_train => n_neuron, Flux.identity),
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
encoder = AutoEncoderToolkit.JointGaussianLogEncoder(
    encoder_chain, µ_layer, logσ_layer
)

## =============================================================================

println("Define SimpleGaussianDecoder...")
# Initialize decoder
decoder = AutoEncoderToolkit.SimpleGaussianDecoder(
    Flux.Chain(
        # First layer
        Flux.Dense(n_latent => n_neuron, Flux.identity),
        # Second Layer
        Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
        # Third layer
        Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
        # Fourth layer
        Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
        # Output layer
        Flux.Dense(n_neuron => n_env_train, Flux.identity)
    )
)

## =============================================================================

# Initialize rhvae
vae = encoder * decoder

## =============================================================================

println("Save model object...")

# Save model object
JLD2.save(
    "$(vae_dir)/vae_model.jld2",
    Dict("model" => vae, "model_state" => Flux.state(vae))
)
