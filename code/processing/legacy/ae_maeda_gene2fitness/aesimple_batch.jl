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
import XLSX

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

# Define number of epochs
n_epoch = 100_000
# Define how often to compute error
n_error = 10_000
# Define batch size
n_batch = 32
# Define number of hidden layers
n_hidden = 4
# Define number of neurons in non-linear hidden layers
n_neuron = [500, 250, 100, 50]
# Define dimensionality of latent space
latent_dim = 3
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

# Check if output directory exists
if !isdir("./output/")
    mkdir("./output/")
end # if

# Check if output directory exists
if !isdir("./output/$(n_epoch)_epoch")
    mkdir("./output/$(n_epoch)_epoch")
end # if

##

# Loading trained AE on fitness data

# Define number of epochs
n_epoch_fit = 10_000_000

# Define filename
fname = "$(git_root())/code/processing/ae_maeda/output/$(n_epoch_fit)_epoch/" *
        "ae_$(latent_dim)dimensions.bson"
# Load model
ae = BSON.load(fname)[:ae]

##

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
IC50_mat = Matrix{Float32}(
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
        IC50_mat[:, i] = Float32.(data.ic50)
    else
        println("group $i stress does not match")
    end # if
end # for

# Fit model to standardize data to mean zero and standard deviation 1 on each
# environment
dt = StatsBase.fit(StatsBase.ZScoreTransform, IC50_mat, dims=2)

# Center data to have mean zero and standard deviation one
IC50_std = StatsBase.transform(dt, IC50_mat);

# Map data to latent space
data_latent = ae.encoder(IC50_std)

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
gene_std = StatsBase.transform(dt, logexpr_mat);

# Define number of genes
n_genes = size(gene_std, 1)

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
encoder_activation_sc = repeat([SimpleChains.relu], length(n_neuron))
encoder_activation_flux = repeat([Flux.relu], n_hidden)

# Define decoder layer and activation function
decoder = n_neuron
decoder_activation_sc = repeat([SimpleChains.relu], length(n_neuron))
decoder_activation_flux = repeat([Flux.relu], length(n_neuron))

##

println("Initializing autoencoder...")

# Initialize list with encoder layers
Encoder = Array{SimpleChains.TurboDense}(undef, length(encoder) + 1)

# Loop through layers
for i = 1:length(encoder)
    # Add layer
    Encoder[i] = SimpleChains.TurboDense(encoder[i], encoder_activation_sc[i])
end # for
# Add last layer from encoder to latent space with activation
Encoder[end] = SimpleChains.TurboDense(latent_dim, latent_activation_sc)

# Initialize autoencoders for each dimensionality
ae_sc = SimpleChains.SimpleChain(
    SimpleChains.static(n_genes),
    Encoder...,
)

# Initialize parameters
param = SimpleChains.init_params(ae_sc)

##

# Initialize full Flux autoencoder
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

# Take encoder as the model to train
ae_flux = ae.encoder

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
ae_loss = SimpleChains.add_loss(ae_sc, SimpleChains.SquaredLoss(data_latent))

# Initialize a gradient matrix, with a number of rows equal to the length
# of the parameter vector p
∇ae_loss = SimpleChains.alloc_threaded_grad(ae_loss);

# Define array where to store errors on training data
ae_mse_sc = Vector{Float32}(undef, n_epoch ÷ n_error + 1)
ae_mse_flux = similar(ae_mse_sc)

# Evaluate initial error
ae_mse_sc[1] = ae_loss(gene_std, param)
# Evaluate error with Flux.jl
ae_mse_flux[1] = Flux.mse(
    Antibiotic.ml.AEs.simple_to_flux(collect(param), ae_flux)(gene_std),
    data_latent
)

# Loop through training ranges
for (j, r) in enumerate(ranges)

    println("Epochs $(r) for $(latent_dim) AE")
    # Train autoencoder
    SimpleChains.train_batched!(
        ∇ae_loss,
        param,
        ae_loss,
        gene_std,
        SimpleChains.ADAM(η_array[j]),
        length(r);
        batchsize=n_batch
    )

    # Evaluate SimpleChains error
    ae_mse_sc[j+1] = ae_loss(gene_std, param)

    # Evaluate Flux MSE
    ae_mse_flux[j+1] = Flux.mse(
        Antibiotic.ml.AEs.simple_to_flux(collect(param), ae_flux)(gene_std),
        data_latent
    )
end # for

BSON.bson(
    fname,
    param=param,
    ae=Antibiotic.ml.AEs.simple_to_flux(collect(param), ae_flux),
    mse_sc=ae_mse_sc,
    mse_flux=ae_mse_flux,
    res_std=IC50_std,
    gene_std=gene_std,
)

println("Done!")