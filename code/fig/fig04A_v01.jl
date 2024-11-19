## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic

# Import AutoEncoderToolkit to train VAEs
import AutoEncoderToolkit as AET

# Import libraries to handle data
import Glob
import DimensionalData as DD
import DataFrames as DF

# Import ML libraries
import Flux

# Import library to save models
import JLD2

# Import basic math
import StatsBase
import Random

# Load Plotting packages
using CairoMakie
using Makie
import ColorSchemes
import Colors
# Activate backend
CairoMakie.activate!()

# Set plotting style
Antibiotic.viz.theme_makie!()

# Set random seed
Random.seed!(42)

## =============================================================================

println("Defining directories...")

# Define version directory
version_dir = "$(git_root())/output/metropolis-hastings_sim/v05"

# Define simulation directory
sim_dir = "$(version_dir)/sim_evo"
# Define VAE directory
vae_dir = "$(version_dir)/vae"
# Define output directory
state_dir = "$(vae_dir)/model_state"

# Define output directory
fig_dir = "$(git_root())/fig/main"

# Generate output directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating output directory...")
    mkpath(fig_dir)
end

## =============================================================================

println("Loading simulation results...")

# Define the subsampling interval
n_sub = 10

# Load fitnotype profiles
fitnotype_profiles = JLD2.load("$(sim_dir)/sim_evo.jld2")["fitnotype_profiles"]

# Extract initial and final time points
t_init, t_final = collect(DD.dims(fitnotype_profiles, :time)[[1, end]])
# Subsample time series
fitnotype_profiles = fitnotype_profiles[time=DD.At(t_init:n_sub:t_final)]

# Define number of environments
n_env = length(DD.dims(fitnotype_profiles, :landscape))

# Extract fitness data bringing the fitness dimension to the first dimension
fit_data = permutedims(fitnotype_profiles.fitness.data, (5, 1, 2, 3, 4, 6))
# Reshape the array to a Matrix
fit_data = reshape(fit_data, size(fit_data, 1), :)

# Reshape the array to stack the 3rd dimension
fit_mat = log.(fit_data)

# Fit model to standardize data to mean zero and standard deviation 1 on each
# environment 
dt = StatsBase.fit(StatsBase.ZScoreTransform, fit_mat, dims=2)

# Standardize the data to have mean 0 and standard deviation 1
log_fitnotype_std = DD.DimArray(
    mapslices(slice -> StatsBase.transform(dt, slice),
        log.(fitnotype_profiles.fitness.data),
        dims=[5]),
    fitnotype_profiles.fitness.dims,
)

## =============================================================================

println("Loading RHVAE model...")

# Find model file
model_file = first(Glob.glob("$(vae_dir)/model*.jld2"[2:end], "/"))
# List epoch parameters
model_states = sort(Glob.glob("$(state_dir)/*.jld2"[2:end], "/"))

# Load model
rhvae = JLD2.load(model_file)["model"]
# Load latest model state
Flux.loadmodel!(rhvae, JLD2.load(model_states[end])["model_state"])
# Update metric parameters
AET.RHVAEs.update_metric!(rhvae)

## =============================================================================

println("Mapping data to latent space...")

# Define latent space dimensions
latent = DD.Dim{:latent}([:latent1, :latent2])

# Map data to latent space
dd_latent = DD.DimArray(
    dropdims(
        mapslices(slice -> rhvae.vae.encoder(slice).μ,
            log_fitnotype_std.data,
            dims=[5]);
        dims=1
    ),
    (log_fitnotype_std.dims[2:4]..., latent, log_fitnotype_std.dims[6]),
)

## =============================================================================

println("Computing Riemannian metric for latent space...")

# Define number of points per axis
n_points = 250

# Extract latent space ranges
latent1_range = range(
    minimum(dd_latent[latent=DD.At(:latent1)]) - 2.5,
    maximum(dd_latent[latent=DD.At(:latent1)]) + 2.5,
    length=n_points
)
latent2_range = range(
    minimum(dd_latent[latent=DD.At(:latent2)]) - 2.5,
    maximum(dd_latent[latent=DD.At(:latent2)]) + 2.5,
    length=n_points
)
# Define latent points to evaluate
z_mat = reduce(hcat, [[x, y] for x in latent1_range, y in latent2_range])

# Compute inverse metric tensor
Ginv = AET.RHVAEs.G_inv(z_mat, rhvae)

# Compute metric 
logdetG = reshape(
    -1 / 2 * AET.utils.slogdet(Ginv), n_points, n_points
)

## =============================================================================

println("Plotting latent space metric...")

# Initialize figure
fig = Figure(size=(500, 300))

# Add global grid layout
gl = GridLayout(fig[1, 1])

# Add grid layout for fig04A section banner
gl04A_banner = gl[1, 1] = GridLayout()
# Add grid layout for Fig04A
gl04A = gl[2, 1] = GridLayout()

# ------------------------------------------------------------------------------

# Add box for section title
Box(
    gl04A_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-40, right=-40) # Moves box to the left and right
)

# Add section title
Label(
    gl04A_banner[1, 1],
    "geometry-informed latent space mapping",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
    alignmode=Mixed(; left=-20) # Moves text to the left
)

# ------------------------------------------------------------------------------

# Add axis
ax1 = Axis(
    gl04A[1, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    title="latent space metric",
    aspect=AxisAspect(1),
    yticklabelsvisible=false,
    xticklabelsvisible=false,
)
ax2 = Axis(
    gl04A[1, 2],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    title="fitness profiles \nlatent coordinates",
    aspect=AxisAspect(1),
    yticklabelsvisible=false,
    xticklabelsvisible=false,
)

# Plot heatmpat of log determinant of metric tensor
hm = heatmap!(
    ax1,
    latent1_range,
    latent2_range,
    logdetG,
    colormap=ColorSchemes.tokyo,
)

heatmap!(
    ax2,
    latent1_range,
    latent2_range,
    logdetG,
    colormap=ColorSchemes.tokyo,
)

# Plot latent space
scatter!(
    ax2,
    vec(dd_latent[latent=DD.At(:latent1)]),
    vec(dd_latent[latent=DD.At(:latent2)]),
    markersize=4,
    color=(:white, 0.3),
)

# Add colorbar
Colorbar(
    gl04A[1, 3],
    hm,
    label="√log[det(G̲̲)]"
)

# Save figure
save("$(fig_dir)/fig04A.pdf", fig)

fig