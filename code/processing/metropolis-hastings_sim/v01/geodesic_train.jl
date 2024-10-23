## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic

# Import project package
import AutoEncoderToolkit as AET
import AutoEncoderToolkit.diffgeo.NeuralGeodesics as NG

# Import libraries to handel data
import DataFrames as DF
import DimensionalData as DD
import Glob

# Import ML libraries
import Flux

# Import library to save models
import JLD2

# Import basic math
import StatsBase
import Random
Random.seed!(42)

## =============================================================================

# Define model hyperparameters

# Define number of time points to evaluate along curve
n_time = 50
# Define number of epochs
n_epoch = 50_000
# Define learning rate
η = 10^-4

## =============================================================================

# Locate current directory
path_dir = pwd()

# Locate current directory
path_dir = pwd()

# Find the path perfix where input data is stored
out_prefix = replace(
    match(r"processing/(.*)", path_dir).match,
    "processing" => "",
)

# Define simulation directory
sim_dir = "$(git_root())/output$(out_prefix)/sim_evo"
# Define model directory
vae_dir = "$(git_root())/output$(out_prefix)/vae"
# Define output directory
state_dir = "$(vae_dir)/model_state"
# Define directory to store trained geodesic curves
geodesic_dir = "$(vae_dir)/geodesic_state/"

# Generate output directory if it doesn't exist
if !isdir(geodesic_dir)
    println("Generating output directory...")
    mkpath(geodesic_dir)
end

## =============================================================================

println("Loading NeuralGeodesic model...")

# Load model template
nng_template = JLD2.load("$(vae_dir)/geodesic.jld2")["model"]

# Load parameters
Flux.loadmodel!(
    nng_template, JLD2.load("$(vae_dir)/geodesic.jld2")["model_state"]
)

# Define time points along curve
t_array = Float32.(collect(range(0, 1, length=n_time)))

## =============================================================================

println("Loading RHVAE model...")

# Load RHVAE model
rhvae = JLD2.load("$(vae_dir)/model.jld2")["model"]
# List parameters for epochs
param_files = sort(Glob.glob("$(state_dir)/*.jld2"[2:end], "/"))
# Load last epoch
Flux.loadmodel!(rhvae, JLD2.load(param_files[end])["model_state"])
# Update metric
AET.RHVAEs.update_metric!(rhvae)

## =============================================================================

println("Loading data into memory...")

# Load fitnotype profiles
fitnotype_profiles = JLD2.load("$(sim_dir)/sim_evo.jld2")["fitnotype_profiles"]

# Reshape the array to stack the 3rd dimension
fit_mat = log.(
    reshape(fitnotype_profiles.fitness.data, size(fitnotype_profiles, 4), :)
)

# Fit model to standardize data to mean zero and standard deviation 1 on each
# environment 
dt = StatsBase.fit(StatsBase.ZScoreTransform, fit_mat, dims=2)

# Standardize the data to have mean 0 and standard deviation 1
log_fitnotype_std = DD.DimArray(
    reduce(
        (x, y) -> cat(x, y, dims=3),
        StatsBase.transform.(
            Ref(dt), eachslice(log.(fitnotype_profiles.fitness.data), dims=3)
        )
    ),
    fitnotype_profiles.fitness.dims,
)

## =============================================================================

println("Map data to latent space...")

# Define latent space dimensions
latent = DD.Dim{:latent}([:latent1, :latent2])

# Map data to latent space
dd_latent = DD.DimArray(
    rhvae.vae.encoder(log_fitnotype_std.data).μ,
    (latent, log_fitnotype_std.dims[2:end]...)
)

## =============================================================================

println("Define lineages over which to train geodesic...")

# Initialize a progress counter
progress_counter = Threads.Atomic{Int}(0)

# Use threading for the loop
Threads.@threads for i in 1:length(DD.dims(dd_latent, :lineage))
    # Update and print progress
    current_progress = Threads.atomic_add!(progress_counter, 1)
    println("Processing lineage: $(current_progress)/$(length(DD.dims(dd_latent, :lineage)))")

    # Find minimum and maximum day
    t_init, t_final = collect(DD.dims(dd_latent, :time)[[1, end]])
    # Define output name
    fname = "$(geodesic_dir)/neuralgeodesic_" *
            "timeinit$(lpad(t_init, 2, "0"))_" *
            "timefinal$(lpad(t_final, 2, "0"))_" *
            "id$(lpad(i, 3, "0"))_" *
            "rhvaeepoch$(lpad(length(param_files), 4, "0"))_" *
            "geoepoch$(lpad(n_epoch, 4, "0")).jld2"

    # Check if file exists
    if isfile(fname)
        continue
    end # if 

    # Extract initial and final points in latent space
    latent_init = dd_latent[time=DD.At(t_init), lineage=DD.At(i)].data
    latent_end = dd_latent[time=DD.At(t_final), lineage=DD.At(i)].data
    # Set NeuralGeodesic model
    nng = NG.NeuralGeodesic(
        deepcopy(nng_template.mlp),
        latent_init,
        latent_end
    )
    # Explicit setup of optimizer
    opt_nng = Flux.Train.setup(
        Flux.Optimisers.Adam(η),
        nng
    )
    # Initialize empty array to save loss
    nng_loss = Vector{Float32}(undef, n_epoch)
    # Loop through epochs
    for epoch in 1:n_epoch
        if epoch % 2_500 == 0
            println("       - Epoch: $(epoch)")
        end
        # Train model and save loss
        nng_loss[epoch] = NG.train!(nng, rhvae, t_array, opt_nng; loss_return=true)
    end # for
    # Save network
    JLD2.jldsave(
        fname,
        model_state=Flux.state(nng),
        latent_init=latent_init,
        latent_end=latent_end,
    )
end # for data in group_vector
