## =============================================================================
println("Loading packages...")

# Import project package
import AutoEncoderToolkit as AET

# Import libraries to handel data
import CSV
import DataFrames as DF
import Glob

# Import ML libraries
import Flux

# Import library to save models
import JLD2

# Import CUDA 
import CUDA

# Import itertools
import IterTools

# Import basic math
import StatsBase
import Random
Random.seed!(42)


## =============================================================================

println("Setting output directories...")

# Define current directory
path_dir = pwd()

# Find the path perfix where input data is stored
prefix = replace(
    match(r"processing/(.*)", path_dir).match,
    "processing" => "",
)

# Define model directory
model_dir = "$(git_root())/output$(prefix)"
# Define model state directory
state_dir = "$(git_root())/output$(prefix)/model_state"
# Define data directory
data_dir = "$(git_root())/output/mcmc_iwasawa_logistic"

## =============================================================================

println("Loading model...")

# Load model
rhvae = JLD2.load("$(model_dir)/model.jld2")["model"]
# List model state files
model_state_files = sort(Glob.glob("$(state_dir)/*.jld2"[2:end], "/"))
# Extract model state file name
model_state_file = model_state_files[end]
# Load model state
model_state = sort(JLD2.load(model_state_file))["model_state"]
# Input parameters to model
Flux.loadmodel!(rhvae, model_state)
# Update metric parameters
AET.RHVAEs.update_metric!(rhvae)

println("Uploading model to GPU...")
# Upload model to GPU
rhvae = rhvae |> Flux.gpu

## =============================================================================

println("Loading data to GPU...")

# Load standardized mean data
data = JLD2.load("$(data_dir)/logic50_preprocess.jld2")["logic50_mcmc_std"]

## =============================================================================

println("Computing MSE...")

# Extract epoch number from model state file name
epoch = match(r"epoch(\d+).jld2", model_state_file).captures[1]

# Define output file name
fname = "$(model_dir)/mcmc_mse_epoch$(epoch).jld2"

# Check if output file exists
if isfile(fname)
    println("Output file already exists. Skipping computation.")
    exit()
end

# Initialize array to store MSE
mse_mcmc = zeros(Float32, size(data, 2), size(data, 3))

for idx in axes(data, 3)
    println("Processing sample $idx/$(size(data, 3))...")

    # Upload batch to GPU
    batch_data_gpu = CUDA.cu(data[:, :, idx])

    # Run through model
    batch_reconstruction = rhvae(batch_data_gpu).μ |> Flux.cpu

    mse_mcmc[:, idx] = Flux.mse.(
        eachcol(batch_reconstruction),
        eachcol(data[:, :, idx]),
    )
end

println("Saving output...")

# Save output
JLD2.save(fname, Dict("mse_mcmc" => mse_mcmc))