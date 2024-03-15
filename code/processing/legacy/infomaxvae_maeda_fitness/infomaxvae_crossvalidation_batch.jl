## 
println("Loading packages...")
# Load project package
@load_pkg Antibiotic

# Import project package
import Antibiotic

# Import AutoEncode package
import AutoEncode

# Import libraries to work with DataFrames
import DataFrames as DF
import CSV

# Import ML libraries
import Flux
import NNlib
import CUDA

# Import library to save models
import BSON

# Import basic math
import Random
import LinearAlgebra
import StatsBase
import Distributions

Random.seed!(42)
##

# Define boolean wether it should be evaluated with gpu
gpu_eval = false

# Define InfoMaxVAE parameters
α = 10.0f0
β = 1.0f0

# Define number of cross-validation events
n_cross = 4
# Define fraction of data for training
train_frac = 0.75
# Define number of epochs
n_epoch = 100_000
# Define how often to compute error
n_error = 1_000
# Define batch size
n_batch = 64
# Define number of hidden layers
n_hidden = 4
# Define number of neurons in non-linear hidden layers
n_neuron = 100
# Define dimensionality of latent space
latent_dim = 3
# Define parameter scheduler
epoch_change = [1, 10^4, 3 * 10^4, 10^5]
learning_rates = [10^-2.75, 10^-3, 10^-3, 10^-3.25]

##
# Check if GPU is available
if gpu_eval & !(CUDA.functional())
    error("GPU is not available")
end # if

##

println("Setting output directories...\n")

# Check if output directory exists
if !isdir("./output/")
    mkdir("./output/")
end # if

# Check if output directory exists
if !isdir("./output/$(n_epoch)_epoch")
    mkdir("./output/$(n_epoch)_epoch")
end # if

##

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
ic50_std = StatsBase.transform(dt, ic50_mat) |> Flux.gpu

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
## Number of cross-validation events
`n_cross = $(n_cross)`
## Fraction of data used for training
`train_frac = $(train_frac)`
## Noise probability
`noise_prob = $(noise_prob)`
## learning rate scheduler
`epoch_change = $(epoch_change)`
`learning_rates = $(learning_rates)`
`AutoEncode.utils.step_scheduler(x, epoch_change, learning_rates)`
"""

# Write README file into memory
open("./output/$(n_epoch)_epoch/README.md", "w") do file
    write(file, readme)
end

##

println("Definining InfoMaxVAE architecture...\n")

#

# Define latent space activation function
latent_activation = Flux.identity
# Define output layer activation function
output_activation = Flux.identity

# Define encoder layer and activation functions
encoder = repeat([n_neuron], n_hidden)
encoder_activation = repeat([Flux.swish], n_hidden)

# Define decoder layer and activation function
decoder = repeat([n_neuron], n_hidden)
decoder_activation = repeat([Flux.swish], n_hidden)

# Define MLP layers and activation functions
mlp = repeat([n_neuron], n_hidden)
mlp_activation = repeat([Flux.swish], n_hidden)

# Define output activation
mlp_output_activation = Flux.identity

##

# Generate splits for reproducibility
train_idx = [
    StatsBase.sample(
        1:n_samples, Int64(round(n_samples * train_frac)), replace=false
    )
    for i = 1:n_cross
]

##

# Loop through cross validation
for cross in 1:n_cross
    println("Training iteration #$(cross) ...\n")
    ##

    println("Setting file name...")
    # Define output filename
    fname = "./output/$(n_epoch)_epoch/" *
            "infomaxvae_$(lpad(cross, 2, "0"))_$(lpad(latent_dim, 2, "0"))dimensions.bson"

    # Check if autoencoder has already been trained
    if isfile(fname)
        println("Iteration #$(cross) with $(latent_dim)D latent space was already processed")
        continue
    end # if

    ##

    # Extract train and test data
    data_train = ic50_std[:, train_idx[cross]]
    data_test = ic50_std[:, setdiff(1:n_samples, train_idx[cross])]

    # Initialize full Flux autoencoder to take advantage of the AEs module.
    vae = AutoEncode.InfoMaxVAEs.infomaxvae_init(
        n_env,
        latent_dim,
        latent_activation,
        output_activation,
        encoder,
        encoder_activation,
        decoder,
        decoder_activation,
        mlp,
        mlp_activation,
        mlp_output_activation,
    )

    ##

    println("Training InfoMaxVAE iteration #$(cross)...")

    # Define array where to store loss function
    vae_loss_train = Float32[]
    vae_loss_test = Float32[]

    # Define array where to store mutual information
    vae_mi_train = Float32[]
    vae_mi_test = Float32[]

    # Define array where to store errors on training data
    vae_mse_train = Float32[]
    # Define array where to store errors on test data
    vae_mse_test = Float32[]


    # Evaluate loss function
    push!(
        vae_loss_train,
        AutoEncode.InfoMaxVAEs.loss(
            vae.vae,
            vae.mlp,
            data_train,
            data_train[:, Random.shuffle(1:end)];
            α=α,
            β=β
        )
    )
    push!(
        vae_loss_test,
        AutoEncode.InfoMaxVAEs.loss(
            vae.vae,
            vae.mlp,
            data_test,
            data_test[:, Random.shuffle(1:end)];
            α=α,
            β=β
        )
    )
    # Evaluate mutual information
    push!(vae_mi_train, AutoEncode.InfoMaxVAEs.mutual_info_mlp(vae, data_train))
    push!(vae_mi_test, AutoEncode.InfoMaxVAEs.mutual_info_mlp(vae, data_test))

    # Evaluate initial error
    push!(vae_mse_train, Flux.cpu(Flux.mse(vae.vae(data_train), data_train)))
    push!(vae_mse_test, Flux.cpu(Flux.mse(vae.vae(data_test), data_test)))

    # Define optimizer to be used for the VAE
    vae_opt = Flux.Train.setup(
        Flux.ADAM(
            AutoEncode.utils.step_scheduler(1, epoch_change, learning_rates)
        ),
        vae.vae
    )

    # Define optimizer to be used for the MLP
    mlp_opt = Flux.Train.setup(
        Flux.ADAM(
            AutoEncode.utils.step_scheduler(1, epoch_change, learning_rates)
        ),
        vae.mlp
    )

    # Loop through epochs
    for j = 1:n_epoch
        # Update optimizer learning rate
        Flux.Optimisers.adjust!(
            vae_opt,
            AutoEncode.utils.step_scheduler(j, epoch_change, learning_rates)
        )
        # Update optimizer learning rate
        Flux.Optimisers.adjust!(
            mlp_opt,
            AutoEncode.utils.step_scheduler(j, epoch_change, learning_rates)
        )

        # Sample data indexes to be used in this epoch
        batch_idx = StatsBase.sample(
            1:Int64(round(n_samples * train_frac)), n_batch, replace=false
        )
        ic50_batch = @view ic50_std[:, batch_idx]

        # Train network
        AutoEncode.InfoMaxVAEs.train!(
            vae,
            ic50_batch,
            vae_opt,
            mlp_opt;
            loss_kwargs=(α=α, β=β)
        )

        # Print progress report
        if (j % n_error == 0)
            println("Epoch #$j for iteration #$(cross)")
            # Evaluate loss function
            push!(
                vae_loss_train,
                Flux.cpu(
                    AutoEncode.InfoMaxVAEs.loss(
                        vae.vae,
                        vae.mlp,
                        data_train,
                        data_train[:, Random.shuffle(1:end)]
                    )
                )
            )

            push!(
                vae_loss_test,
                Flux.cpu(
                    AutoEncode.InfoMaxVAEs.loss(
                        vae.vae,
                        vae.mlp,
                        data_test,
                        data_test[:, Random.shuffle(1:end)],
                        α=α,
                        β=β,
                    )
                )
            )

            # Evaluate mutual information
            push!(
                vae_mi_train,
                AutoEncode.InfoMaxVAEs.mutual_info_mlp(vae, data_train)
            )
            push!(
                vae_mi_test,
                AutoEncode.InfoMaxVAEs.mutual_info_mlp(vae, data_test)
            )

            # Evaluate error
            push!(vae_mse_train, Flux.cpu(Flux.mse(vae.vae(data_train), data_train)))
            push!(vae_mse_test, Flux.cpu(Flux.mse(vae.vae(data_test), data_test)))

            println("Loss train: $(vae_loss_train[end])")
            println("Loss test: $(vae_loss_test[end])\n")

            println("Mutual Info train: $(vae_mi_train[end])")
            println("Mutual Info test: $(vae_mi_test[end])\n")

            println("MSE train: $(vae_mse_train[end])")
            println("MSE test: $(vae_mse_test[end])\n")
        end # if
    end # for

    println("Saving data into memory...\n")

    BSON.bson(
        fname,
        ae=Flux.cpu(vae),
        mse_train=vae_mse_train,
        mse_test=vae_mse_test,
        mi_train=vae_mi_train,
        mi_test=vae_mi_test,
        loss_train=vae_loss_train,
        loss_test=vae_loss_test,
        data_train=Flux.cpu(data_train),
        data_test=Flux.cpu(data_test),
    )

    println("Done with iteration #$(cross)!")
end # for


println("Done!")