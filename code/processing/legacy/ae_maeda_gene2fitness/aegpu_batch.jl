## 
println("Loading packages...")
# Load project package
@load_pkg Antibiotic

# Import project package
import Antibiotic

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

Random.seed!(42)
##

# Check if GPU is available
if !(CUDA.functional())
    error("GPU is not available")
end # if

# Check CUDA information
CUDA.versioninfo()

##

# Define number of epochs
n_epoch = 100_000
# Define how often to compute error
n_error = 1_000
# Define batch size
n_batch = 32
# Define number of neurons in non-linear hidden layers
n_neuron = [1000, 500, 250, 125, 75]
# Define dimensionality of latent space
latent_dim = 3

# Define parameter scheduler
epoch_change = [1, 4_000, 10_000, 50_000, 10_000]
learning_rates = [10^-5, 10^-6, 10^-6.5, 10^-6, 10^-6.5];

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

# Define output filename
fname = "./output/$(n_epoch)_epoch/ae_$(latent_dim)dimensions.bson"

# Check if autoencoder has already been trained
if isfile(fname)
    error("$(latent_dim)-D was already processed")
end # if

##

println("Loading AE trained on fitness data...\n")

# Define number of epochs
n_epoch_fit = 10_000_000

# Define filename
ae_name = "$(git_root())/code/processing/ae_maeda/output/$(n_epoch_fit)_epoch/" *
          "ae_$(latent_dim)dimensions.bson"
# Load model
ae_fit = BSON.load(ae_name)[:ae]

##

println("Loading data...\n")

# Define data directory
data_dir = "$(git_root())/data/Maeda_2020"

# Load resistance data
df_res = CSV.read("$(data_dir)/maeda_resistance_tidy.csv", DF.DataFrame)

# Load gene expression data
df_gene = CSV.read("$(data_dir)/maeda_logexpr_tidy.csv", DF.DataFrame)
# Remove ancestral strain data
df_gene = df_gene[.!(occursin.("MDS", df_gene.strain)), :]

# Sort dataframes by strain
DF.sort!(df_res, :strain)
DF.sort!(df_gene, :strain)

##

# Group data by strain
gene_group = DF.groupby(df_gene, :strain, sort=true)
res_group = DF.groupby(df_res, :strain, sort=true)

# Extract groups to compare with groups from df_res
gene_groups = [x[:strain] for x in keys(gene_group)]
res_groups = [x[:strain] for x in keys(res_group)]

# Stop script if group order doesnt match
if !all(gene_groups .== res_groups)
    error("Grouped dataframes don't follow same order")
end # if

##

# Initialize array to save data
ic50_mat = Matrix{Float32}(
    undef, length(unique(df_res.stress)), length(unique(df_res.strain))
)

# Extract unique stresses to make sure the matrix is built correctly
stress = sort(unique(df_res.stress))

# Loop through groups
for (i, data) in enumerate(res_group)
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

# Fit model to standardize data to mean zero and standard deviation 1 on each
# environment
dt = StatsBase.fit(StatsBase.ZScoreTransform, ic50_mat, dims=2)

# Center data to have mean zero and standard deviation one
ic50_std = StatsBase.transform(dt, ic50_mat);

# Map data to latent space
ic50_latent = ae_fit.encoder(ic50_std) |> Flux.gpu

##

# Initialize array to save data
logexpr_mat = Matrix{Float32}(
    undef, length(unique(df_gene.gene)), length(unique(df_gene.strain))
)

# Extract unique gene names to make sure the matrix is built correctly
genes = sort(unique(df_gene.gene))

# Loop through groups
for (i, data) in enumerate(gene_group)
    # Sort data by stress
    DF.sort!(data, :gene)
    # Check that the stress are in the correct order
    if all(data.gene .== genes)
        # Add data to matrix
        logexpr_mat[:, i] = Float32.(data.logΔexpr)
    else
        println("group $i genes does not match")
    end # if
end # for

# Fit model to standardize data to mean zero and standard deviation 1 on each
# environment
dt = StatsBase.fit(StatsBase.ZScoreTransform, logexpr_mat, dims=2)

# Center data to have mean zero and standard deviation one
gene_std = StatsBase.transform(dt, logexpr_mat) |> Flux.gpu

# Define number of genes
n_genes = size(gene_std, 1)
# Define number of samples
n_samples = size(gene_std, 2)

##

println("Definining encoder architecture...\n")

# Define latent space activation function
latent_activation_flux = Flux.identity
# Define output layer activation function
output_activation_flux = Flux.identity

# Define encoder layer and activation functions
encoder = n_neuron
encoder_activation_flux = repeat([Flux.relu], length(n_neuron))

# Define decoder layer and activation function
decoder = n_neuron
decoder_activation_flux = repeat([Flux.relu], length(n_neuron))

##

# Initialize full Flux autoencoder to take advantage of the AEs module. Note: we
# will only train the encoder section since the output should be the learned
# latent space.
ae = Antibiotic.ml.AEs.ae_init(
    n_genes,
    latent_dim,
    latent_activation_flux,
    output_activation_flux,
    encoder,
    encoder_activation_flux,
    decoder,
    decoder_activation_flux,
)

# Take encoder as the model to train and upload to GPU
ae_flux = ae.encoder |> Flux.gpu

##

println("Definining loss function as MSE...")
# Define loss function. The loss function is the MSE between the encoder output
# and the learned latent space built from the fitness data only
loss(x, y) = Flux.mse(ae_flux(x), y)

##

println("Training encoder...")

# Define array where to store errors on training data
ae_mse = Vector{Float32}(undef, n_epoch ÷ n_error + 1)
# Evaluate initial error
ae_mse[1] = loss(gene_std, ic50_latent)

# Initialize error counter
global mse_counter = 2

# Loop through epochs
for j = 1:n_epoch
    # Sample data indexes to be used in this epoch
    batch_idx = StatsBase.sample(
        1:n_samples, n_batch, replace=false
    )
    # Extract minibatch gene expression data
    gene_batch = gene_std[:, batch_idx]
    # Extract minibatch latent spaace code data
    latent_batch = ic50_latent[:, batch_idx]

    # Train network
    Flux.train!(
        loss,
        Flux.params(ae_flux),
        zip(eachcol(gene_batch), eachcol(latent_batch)),
        Flux.ADAM(
            Antibiotic.ml.step_scheduler(j, epoch_change, learning_rates)
        )
    )

    # Print progress report
    if (j % n_error == 0)
        println("Epoch # $j")
        ae_mse[mse_counter] = loss(gene_batch, latent_batch) |> Flux.cpu

        println("MSE: $(ae_mse[mse_counter])")

        # Update counter
        global mse_counter += 1
    end # if
end # for

println("Saving data into memory...\n")

BSON.bson(
    fname,
    ae=Flux.cpu(ae_flux),
    mse=ae_mse,
    ic50_latent=Flux.cpu(ic50_latent),
    gene_std=Flux.cpu(gene_std),
)

println("Done!")
