## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic
import Antibiotic.mh as mh

# Import AutoEncoderToolkit to train VAEs
import AutoEncoderToolkit as AET

# Import libraries to handle data
import Glob
import CSV
import DataFrames as DF
import DimensionalData as DD

# Import ML libraries
import Flux

# Import library to save models
import JLD2

# Import IterTools for Cartesian product
import IterTools

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
# Define directory to store trained geodesic curves
geodesic_dir = "$(vae_dir)/geodesic_state/"
# Define output directory
fig_dir = "$(git_root())/fig/main"

# Generate output directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating output directory...")
    mkpath(fig_dir)
end

## =============================================================================

println("Loading simulation landscapes...")

# Load fitness landscapes
fitness_landscapes = JLD2.load("$(sim_dir)/sim_evo.jld2")["fitness_landscapes"]
# Load mutational landscape
genetic_density = JLD2.load(
    "$(sim_dir)/sim_evo.jld2"
)["genetic_density"]

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

# Reorder dimensions
dd_latent = permutedims(dd_latent, (4, 1, 2, 3, 5))

## =============================================================================

println("Computing Riemannian metric for latent space...")

# Define number of points per axis
n_points = 100

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

# Convert product to array of vectors
latent_grid = [
    Float32.([x, y]) for (x, y) in IterTools.product(latent1_range, latent2_range)
]

# Define mask for fitness landscape
mask = (maximum(logdetG) * 0.90 .< logdetG .≤ maximum(logdetG))

## =============================================================================

println("Listing geodesic curves...")

# List all files in the directory
geodesic_files = Glob.glob("$(geodesic_dir)/*.jld2"[2:end], "/")

# Initialize dataframe to store files metadata
df_meta = DF.DataFrame()

# Loop over geodesic state files
for gf in geodesic_files
    # Extract initial generation number from file name using regular expression
    t_init = parse(Int, match(r"timeinit(\d+)", gf).captures[1])
    # Extract final generation number from file name using regular expression
    t_final = parse(Int, match(r"timefinal(\d+)", gf).captures[1])
    # Extract lineage number from file name using regular expression
    lin = parse(Int, match(r"lineage(\d+)", gf).captures[1])
    # Extract replicate number from file name using regular expression
    rep = parse(Int, match(r"replicate(\d+)", gf).captures[1])
    # Extract evolution condition from file name using regular expression
    evo = parse(Int, match(r"evo(\d+)", gf).captures[1])
    # Extract RHVAE epoch number from file name using regular expression
    rhvae_epoch = parse(Int, match(r"rhvaeepoch(\d+)", gf).captures[1])
    # Extract geodesic epoch number from file name using regular expression
    geo_epoch = parse(Int, match(r"geoepoch(\d+)", gf).captures[1])
    # Append as DataFrame
    DF.append!(
        df_meta,
        DF.DataFrame(
            :t_init => t_init,
            :t_final => t_final,
            :lineage => lin,
            :rep => rep,
            :evo => evo,
            :rhvae_epoch => rhvae_epoch,
            :geodesic_epoch => geo_epoch,
            :geodesic_state => gf,
        ),
    )
end # for gf in geodesic_files

## =============================================================================

println("Plotting example geodesic curves...")

# Define environment indexes
env_idxs = [1, 8, 25, 36, 23, 24]

# Define number of rows and columns
rows = 2
cols = 3

# Initialize figure
fig = Figure(size=(160 * cols, 200 * rows))

# Add global grid layout
gl = GridLayout(fig[1, 1])

# Add grid layout for title banner
gl04D_banner = GridLayout(gl[1, 1])
# Add grid layout for plots
gl04D = GridLayout(gl[2, 1])

# ------------------------------------------------------------------------------
# Add labels
# ------------------------------------------------------------------------------

# Add box for section title
Box(
    gl04D_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-20, right=0),
    tellwidth=false,
    tellheight=false,
)
# Title
Label(
    gl04D_banner[1, 1],
    "example geodesic curves in latent space",
    padding=(0, 0, 0, 0),
    halign=:left,
    alignmode=Mixed(; left=0),
    tellwidth=false,
    tellheight=true,
)

# ------------------------------------------------------------------------------
# Loop over fitness landscapes
for (i, env_idx) in enumerate(env_idxs)
    # Define row and column
    row = (i - 1) ÷ cols + 1
    col = (i - 1) % cols + 1
    # Add axis for latent fitness landscape
    ax_latent = Axis(gl04D[row, col], aspect=AxisAspect(1))
    # Remove axis labels
    hidedecorations!(ax_latent)

    # --------------------------------------------------------------------------
    # Plot latent fitness landscape
    # --------------------------------------------------------------------------

    # Map latent grid to output space
    F_latent = getindex.(
        getindex.(rhvae.vae.decoder.(latent_grid), :μ),
        env_idx
    )

    # Apply mask
    F_latent_masked = (mask .* minimum(F_latent)) .+ (F_latent .* .!mask)

    # Plot latent fitness landscape
    heatmap!(
        ax_latent,
        latent1_range,
        latent2_range,
        F_latent_masked,
        colormap=:algae,
    )

    # Plot latent fitness landscape contour lines
    contour!(
        ax_latent,
        latent1_range,
        latent2_range,
        F_latent_masked,
        color=:black,
        linestyle=:dash,
        levels=7
    )

    # --------------------------------------------------------------------------
    # Plot example trajectories
    # --------------------------------------------------------------------------
    # Extract data for environment
    latent_data = dd_latent[landscape=1, evo=env_idx, replicate=1]

    # Initialize counter
    counter = 1
    # Loop over lineages
    for lin in DD.dims(latent_data, :lineage)
        # Extract trajectory
        scatterlines!(
            ax_latent,
            latent_data[lineage=lin, latent=DD.At(:latent1)].data,
            latent_data[lineage=lin, latent=DD.At(:latent2)].data,
            color=ColorSchemes.glasbey_hv_n256[counter],
            linewidth=1.5,
            markersize=3,
        )
        # Increment counter
        counter += 1
    end # for lin

end # for

# Add global axis labels
Label(
    gl04D[end, :, Bottom()],
    "latent dimension 1",
    fontsize=16,
    padding=(0, 0, 0, 0),
    tellwidth=false,
    tellheight=true,
)
Label(
    gl04D[:, 1, Left()],
    "latent dimension 2",
    fontsize=16,
    rotation=π / 2,
    padding=(0, 0, 0, 0),
    tellwidth=true,
    tellheight=true,
)

fig