## 
println("Loading packages...")
# Load project package
@load_pkg Antibiotic

# Import package to revise module
import Revise
# Import project package
import Antibiotic

# Import library to list files
import Glob

# Import libraries to work with DataFrames
import DataFrames as DF
import CSV

# Import ML libraries
import Flux
import NNlib
import SimpleChains

# Import library to save models
import BSON

# Import basic math
import Random
import LinearAlgebra
import StatsBase

Random.seed!(42)
##

println("Reading metadata...")

# Define number of epochs
n_epoch = 50_000_000
# Define how often to compute error
n_error = 10_000
# Define batch size
n_batch = 32
# Define number of hidden layers
n_hidden = 4
# Define number of neurons in non-linear hidden layers
n_neuron = 100
# Define dimensionality of latent space
latent_dim = 2
# Define parameter scheduler
epoch_change = [1, 10^4, 10^5, 5 * 10^5, 10^6]
learning_rates = [10^-4, 10^-5, 10^-5.5, 10^-6, 10^-7];

##

# Define positions for each of the ranges to evaluate
steps = collect(1:n_error:n_epoch)

# List ranges for each iteration
ranges = [steps[i]:steps[i+1]-1 for i = 1:length(steps)-1]
# If needed, add last step
if maximum(ranges[end]) < n_epoch
    push!(ranges, maximum(ranges[end])+1:n_epoch)
end # if

# Define learning rates
η_array = [
    Antibiotic.ml.step_scheduler(x, epoch_change, learning_rates)
    for x in minimum.(ranges)
]

##

println("Loading data into memory...")

# Define data directory
data_dir = "$(git_root())/data/Maeda_2020"

# Load file into memory
df_res = CSV.read("$(data_dir)/maeda_resistance_tidy.csv", DF.DataFrame)

# Initialize array to save data
IC50_mat = Matrix{Float32}(
    undef, length(unique(df_res.stress)), length(unique(df_res.strain))
)
# Group data by strain
df_group = DF.groupby(df_res, :strain)

# Extract unique stresses to make sure the matrix is built correctly
stress = sort(unique(df_res.stress))

# Loop through groups
for (i, data) in enumerate(df_group)
    # Sort data by stress
    DF.sort!(data, :stress)
    # Check that the stress are in the correct order
    if all(data.stress .== stress)
        # Add data to matrix
        IC50_mat[:, i] = Float32.(data.ic50)
    else
        println("group $i stress does not match")
    end # if
end # for

# Define number of environments
n_env = size(IC50_mat, 1)

# Fit model to standardize data to mean zero and standard deviation 1 on each
# environment
dt = StatsBase.fit(StatsBase.ZScoreTransform, IC50_mat, dims=2)

# Center data to have mean zero and standard deviation one
IC50_std = StatsBase.transform(dt, IC50_mat);

##

# Check if output directory exists
if !isdir("./output/")
    mkdir("./output/")
end # if

# Check if output directory exists
if !isdir("./output/$(n_epoch)_epoch")
    mkdir("./output/$(n_epoch)_epoch")
end # if

##

println("Writing down metadata to README.md file")

# Define text to go into README
readme = """
# `$(@__FILE__)`
## Latent space dimensionalities to train
`latent_dim = $(latent_dim)`
## Number of epochs
`n_epoch = $(n_epoch)`
## Number of epochs between error evaluations
`n_error = $(n_error)`
## Batch size
`n_batch = $(n_batch)`
## Number of hidden non-linear layers
`n_hidden = $(n_hidden)`
## Number of neurons per layer
`n_neuron = $(n_neuron)`
## learning rate scheduler
`epoch_change = $(epoch_change)`
learning_rates = $(learning_rates)`
`Antibiotic.ml.step_scheduler(x, epoch_change, learning_rates)`
"""

# Write README file into memory
open("./output/$(n_epoch)_epoch/README.md", "w") do file
    write(file, readme)
end

##

# Define autoencoder architecture

# Define latent space activation function
latent_activation_sc = SimpleChains.identity
latent_activation_flux = Flux.identity
# Define output layer activation function
output_activation_sc = SimpleChains.identity
output_activation_flux = Flux.identity

# Define encoder layer and activation functions
encoder = repeat([n_neuron], n_hidden)
encoder_activation_sc = repeat([SimpleChains.relu], n_hidden)
encoder_activation_flux = repeat([Flux.relu], n_hidden)

# Define decoder layer and activation function
decoder = repeat([n_neuron], n_hidden)
decoder_activation_sc = repeat([SimpleChains.relu], n_hidden)
decoder_activation_flux = repeat([Flux.relu], n_hidden)

##

println("Initializing autoencoder...")
# Initialize autoencoders for each dimensionality
ae_sc = Antibiotic.ml.AEs.SimpleAutoencoder(
    n_env,
    latent_dim,
    latent_activation_sc,
    output_activation_sc,
    encoder,
    encoder_activation_sc,
    decoder,
    decoder_activation_sc,
)
# Initialize parameters
param = SimpleChains.init_params(ae_sc)

ae_flux = Antibiotic.ml.AEs.Autoencoder(
    n_env,
    latent_dim,
    latent_activation_flux,
    output_activation_flux,
    encoder,
    encoder_activation_flux,
    decoder,
    decoder_activation_flux,
)

##

println("Setting output directories...")
# Define output filename
fname = "./output/$(n_epoch)_epoch/ae_$(latent_dim)dimensions.bson"

# Check if autoencoder has already been trained
if isfile(fname)
    error("$(latent_dim)-D was already processed")
end # if

##

println("Training AE...")

# Initialize autoencoder with loss function
ae_loss = SimpleChains.add_loss(ae_sc, SimpleChains.SquaredLoss(IC50_std))

# Initialize a gradient matrix, with a number of rows equal to the length
# of the parameter vector p
∇ae_loss = SimpleChains.alloc_threaded_grad(ae_loss);

# Define array where to store errors on training data
ae_mse_sc = Vector{Float32}(undef, n_epoch ÷ n_error + 1)
ae_mse_flux = similar(ae_mse_sc)

# Evaluate initial error
ae_mse_sc[1] = ae_loss(IC50_std, param)
ae_mse_flux[1] = Flux.mse(
    Antibiotic.ml.AEs.simple_to_flux(collect(param), ae_flux)(IC50_std),
    IC50_std
)

# Loop through training ranges
for (j, r) in enumerate(ranges)

    println("Epochs $(r) for $(latent_dim) AE")
    # Train autoencoder
    SimpleChains.train_batched!(
        ∇ae_loss,
        param,
        ae_loss,
        IC50_std,
        SimpleChains.ADAM(η_array[j]),
        length(r);
        batchsize=n_batch
    )

    # Evaluate SimpleChains error
    ae_mse_sc[j+1] = ae_loss(IC50_std, param)

    # Evaluate Flux MSE
    ae_mse_flux[j+1] = Flux.mse(
        Antibiotic.ml.AEs.simple_to_flux(
            collect(param), ae_flux
        )(IC50_std),
        IC50_std
    )
end # for

BSON.bson(
    fname,
    ae=Antibiotic.ml.AEs.simple_to_flux(collect(param), ae_flux),
    mse_sc=ae_mse_sc,
    mse_flux=ae_mse_flux,
    data=IC50_mat,
    data_std=IC50_std,
)

println("Done!")
