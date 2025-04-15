## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic
import Antibiotic.mh as mh

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

# Import IterTools for Cartesian product
import IterTools

# Import basic math
import LinearAlgebra
import MultivariateStats as MStats
import StatsBase
import Random
import Distances

# Load Plotting packages
using CairoMakie
using Makie
using Makie.StructArrays
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
# Define output directory
rhvae_state_dir = "$(vae_dir)/model_state"
vae_state_dir = "$(vae_dir)/vae_model_state"
# Define output directory
fig_dir = "$(git_root())/fig/supplementary"

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

# Find model file
model_file = first(Glob.glob("$(vae_dir)/model*.jld2"[2:end], "/"))
# List RHVAE epoch parameters
rhvae_model_states = sort(Glob.glob("$(rhvae_state_dir)/*.jld2"[2:end], "/"))
# List VAE epoch parameters
vae_model_states = sort(Glob.glob("$(vae_state_dir)/*.jld2"[2:end], "/"))

# Load model
rhvae = JLD2.load(model_file)["model"]
# Load latest model state
Flux.loadmodel!(rhvae, JLD2.load(rhvae_model_states[end])["model_state"])
# Update metric parameters
AET.RHVAEs.update_metric!(rhvae)

# Load VAE model
vae = JLD2.load("$(vae_dir)/vae_model.jld2")["model"]
# Load latest model state
Flux.loadmodel!(vae, JLD2.load(vae_model_states[end])["model_state"])

## =============================================================================

println("Loading data into memory...")

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

# Standardize entire matrix
fit_mat_std = StatsBase.transform(dt, fit_mat)

## =============================================================================

println("Map data to RHVAE latent space...")

# Define latent space dimensions
latent = DD.Dim{:latent}([:latent1, :latent2])

# Map data to latent space
dd_rhvae_latent = DD.DimArray(
    dropdims(
        mapslices(slice -> rhvae.vae.encoder(slice).μ,
            log_fitnotype_std.data,
            dims=[5]);
        dims=1
    ),
    (log_fitnotype_std.dims[2:4]..., latent, log_fitnotype_std.dims[6]),
)

## =============================================================================

println("Map data to VAE latent space...")

# Map data to latent space
dd_vae_latent = DD.DimArray(
    dropdims(
        mapslices(slice -> vae.encoder(slice).μ,
            log_fitnotype_std.data,
            dims=[5]);
        dims=1
    ),
    (log_fitnotype_std.dims[2:4]..., latent, log_fitnotype_std.dims[6]),
)

## =============================================================================

println("Performing PCA on the data...")

# Perform PCA on the data 
fit_pca = MStats.fit(MStats.PCA, fit_mat_std, maxoutdim=2)

# Define latent space dimensions
pca_dims = DD.Dim{:pca}([:pc1, :pc2])

# Map data to latent space
dd_pca = DD.DimArray(
    dropdims(
        mapslices(slice -> MStats.predict(fit_pca, slice),
            log_fitnotype_std.data,
            dims=[5]);
        dims=1
    ),
    (log_fitnotype_std.dims[2:4]..., pca_dims, log_fitnotype_std.dims[6]),
)

## =============================================================================

println("Joining data into single structure...")
# Extract phenotype data
dd_phenotype = fitnotype_profiles.phenotype[landscape=DD.At(1)]

# Join phenotype, latent and PCA space data all with the same dimensions
dd_join = DD.DimStack(
    (
    phenotype=dd_phenotype,
    rhvae=permutedims(dd_rhvae_latent, (4, 1, 2, 3, 5)),
    vae=permutedims(dd_vae_latent, (4, 1, 2, 3, 5)),
    pca=permutedims(dd_pca, (4, 1, 2, 3, 5)),
),
)

## =============================================================================

println("Computing Z-score transforms...")

# Extract data matrices and standardize data to mean zero and standard deviation 1
# and standard deviation 1
dt_dict = Dict(
    :phenotype => StatsBase.fit(
        StatsBase.ZScoreTransform,
        reshape(dd_join.phenotype.data, 2, :),
        dims=2
    ),
    :rhvae => StatsBase.fit(
        StatsBase.ZScoreTransform,
        reshape(dd_join.rhvae.data, 2, :),
        dims=2
    ),
    :vae => StatsBase.fit(
        StatsBase.ZScoreTransform,
        reshape(dd_join.vae.data, 2, :),
        dims=2
    ),
    :pca => StatsBase.fit(
        StatsBase.ZScoreTransform,
        reshape(dd_join.pca.data, 2, :),
        dims=2
    ),
)

## =============================================================================

println("Computing Euclidean distances with respect to phenotype space...")

# Compute all pairwise distances
dist_dict = Dict(
    :phenotype => Distances.pairwise(
        Distances.Euclidean(),
        StatsBase.transform(
            dt_dict[:phenotype], reshape(dd_join.phenotype.data, 2, :)
        )
    ),
    :rhvae => Distances.pairwise(
        Distances.Euclidean(),
        StatsBase.transform(dt_dict[:rhvae], reshape(dd_join.rhvae.data, 2, :))
    ),
    :vae => Distances.pairwise(
        Distances.Euclidean(),
        StatsBase.transform(dt_dict[:vae], reshape(dd_join.vae.data, 2, :))
    ),
    :pca => Distances.pairwise(
        Distances.Euclidean(),
        StatsBase.transform(dt_dict[:pca], reshape(dd_join.pca.data, 2, :))
    ),
)

## =============================================================================

println("Plotting Euclidean distances...")

# Initialize figure
fig = Figure(size=(900, 350))

# Add global grid layout
gl = GridLayout(fig[1, 1])

# Add grid layout for section banner
gl_banner = gl[1, 1] = GridLayout()
# Add grid layout for Fig04A
gl_fig = gl[2, 1] = GridLayout()

# ------------------------------------------------------------------------------
# Add section banner
# ------------------------------------------------------------------------------

# Add box for section title
Box(
    gl_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-35, right=0), # Moves box to the left and right
)

# Add section title
Label(
    gl_banner[1, 1],
    "pairwise Euclidean distances comparison",
    fontsize=14,
    padding=(-25, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
)

# ------------------------------------------------------------------------------
# Plot Euclidean distances
# ------------------------------------------------------------------------------

# Extract phenotype-space lower triangular matrix
dist_mat = dist_dict[:phenotype]
dist_phenotype = dist_mat[LinearAlgebra.tril(trues(size(dist_mat)), -1)]

cmap = to_colormap(:BuPu_9)
cmap[1] = RGBAf(1, 1, 1, 1) # make sure background is white

# Loop through latent spaces
for (i, space) in enumerate([:pca, :vae, :rhvae])
    # Extract lower triangular matrix
    dist_mat = dist_dict[space]
    dist_space = dist_mat[LinearAlgebra.tril(trues(size(dist_mat)), -1)]
    # Compute R-squared
    r2 = 1 -
         sum((dist_phenotype .- dist_space) .^ 2) /
         sum((dist_phenotype .- StatsBase.mean(dist_phenotype)) .^ 2)
    # Convert points to StructArray
    points = StructArray{Point2f}((dist_phenotype, dist_space))
    # Initialize axis
    ax = Axis(
        gl_fig[1, i],
        title="$(uppercase(string(space))) | R² = $(round(r2[1], digits=2))",
        xlabel="phenotype-space distance",
        ylabel="latent-space distance",
        aspect=AxisAspect(1),
    )
    # Plot distance scatter plot
    datashader!(ax, points, colormap=cmap, async=false)
    # Add diagonal line
    lines!(
        ax,
        [0, maximum(dist_phenotype)],
        [0, maximum(dist_phenotype)],
        color=:black,
        linewidth=2,
        linestyle=:dash,
    )
    # Set limits
    xlims!(ax, (0, 6))
    ylims!(ax, (0, 6))
end

# Save figure
save("$(fig_dir)/figSI_euclidean-distance-comparison.pdf", fig)
save("$(fig_dir)/figSI_euclidean-distance-comparison.png", fig)

# Display figure
fig

## =============================================================================