## 
println("Loading packages...")
# Load project package
@load_pkg Antibiotic

# Import package to revise module
import Revise
# Import project package
import Antibiotic

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

# Import library to access classic ML datasets
import MLDatasets

Random.seed!(42)
##

println("Reading metadata...")

# Define number of MNIST entries to keep per digit
n_data = 1000
# Define number of epochs
n_epoch = 500_000
# Define how often to compute error
n_error = 10_000
# Define batch size
n_batch = 32
# Define number of hidden layers
n_hidden = 2
# Define number of neurons in non-linear hidden layers
n_neuron = [64, 32]
# Define dimensionality of latent space
latent_dim = 2
# Define parameter scheduler
epoch_change = [1, 3 * 10^3, 6 * 10^3, 10^4, 5 * 10^4]
learning_rates = [10^-5, 10^-6, 10^-6.5, 10^-7, 10^-7.5];

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

# Access the data
data = MLDatasets.MNIST(:train)

# Find index of zeros or ones
idx_zero = (data.targets .== 0)
idx_one = (data.targets .== 1)
# Extract zeros and ones
data_zero = data.features[:, :, idx_zero][:, :, 1:n_data]
data_one = data.features[:, :, idx_one][:, :, 1:n_data];

# Extract features into long 1D arrays
zero_mat = hcat([vec(data_zero[:, :, i]) for i = 1:n_data]...)
one_mat = hcat([vec(data_one[:, :, i]) for i = 1:n_data]...)
# Compile all data into a single matrix
data_mat = hcat([zero_mat, one_mat]...)

# Rescale the data to be ∈ [-1, 1]
scale_values = [-1.0, 1.0]
scale_range = maximum(scale_values) - minimum(scale_values)
data_range = maximum(data_mat) - minimum(data_mat)
data_std = Float32.((data_mat .- data_range / 2) .* scale_range ./ data_range)

# Define number of inputs
n_input = size(data_std, 1)

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
encoder = n_neuron
encoder_activation_sc = repeat([SimpleChains.softplus], n_hidden)
encoder_activation_flux = repeat([Flux.softplus], n_hidden)

# Define decoder layer and activation function
decoder = reverse(n_neuron)
decoder_activation_sc = repeat([SimpleChains.softplus], n_hidden)
decoder_activation_flux = repeat([Flux.softplus], n_hidden)

##

println("Initializing autoencoder...")
# Initialize autoencoders for each dimensionality
ae_sc = Antibiotic.ml.AEs.SimpleAutoencoder(
    n_input,
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
    n_input,
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
ae_loss = SimpleChains.add_loss(ae_sc, SimpleChains.SquaredLoss(data_std))

# Initialize a gradient matrix, with a number of rows equal to the length
# of the parameter vector p
∇ae_loss = SimpleChains.alloc_threaded_grad(ae_loss);

# Define array where to store errors on training data
ae_mse_sc = Vector{Float32}(undef, n_epoch ÷ n_error + 1)
ae_mse_flux = similar(ae_mse_sc)

# Evaluate initial error
ae_mse_sc[1] = ae_loss(data_std, param)
ae_mse_flux[1] = Flux.mse(
    Antibiotic.ml.AEs.simple_to_flux(collect(param), ae_flux)(data_std),
    data_std
)

# Loop through training ranges
for (j, r) in enumerate(ranges)

    println("Epochs $(r) for $(latent_dim) AE")
    # Train autoencoder
    SimpleChains.train_batched!(
        ∇ae_loss,
        param,
        ae_loss,
        data_std,
        SimpleChains.ADAM(η_array[j]),
        length(r);
        batchsize=n_batch
    )

    # Evaluate SimpleChains error
    ae_mse_sc[j+1] = ae_loss(data_std, param)

    # Evaluate Flux MSE
    ae_mse_flux[j+1] = Flux.mse(
        Antibiotic.ml.AEs.simple_to_flux(
            collect(param), ae_flux
        )(data_std),
        data_std
    )

    # Save partial process
    if (maximum(r) % 50_000) == 0
        println("Saving partial progress for $(maximum(r)) epoch")
        BSON.bson(
            "./output/$(n_epoch)_epoch/ae_$(latent_dim)dimensions" *
            "_$(maximum(r))partial.bson",
            ae=Antibiotic.ml.AEs.simple_to_flux(collect(param), ae_flux),
            mse_sc=ae_mse_sc[1:j+1],
            mse_flux=ae_mse_flux[1:j+1],
        )
    end # if
end # for

BSON.bson(
    fname,
    ae=Antibiotic.ml.AEs.simple_to_flux(collect(param), ae_flux),
    mse_sc=ae_mse_sc,
    mse_flux=ae_mse_flux,
    data=data_mat,
    data_std=data_std,
)

println("Done!")
