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
# Define directory for neural network
mlp_dir = "$(version_dir)/mlp"
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

println("Loading simulation landscapes...")

# Load fitness landscapes
fitness_landscapes = JLD2.load("$(sim_dir)/sim_evo.jld2")["fitness_landscapes"]

# Load mutational landscape
genetic_density = JLD2.load("$(sim_dir)/sim_evo.jld2")["genetic_density"]

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
n_points = 75

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

# Define limits of phenotype space
phenotype_lims = (
    x=(-5, 5),
    y=(-5, 5),
)

# Define range of phenotypes to evaluate
pheno1 = range(phenotype_lims.x..., length=n_points)
pheno2 = range(phenotype_lims.y..., length=n_points)

# Create meshgrid for genetic density
G = mh.genetic_density(pheno1, pheno2, genetic_density)

# Convert product to array of vectors
latent_grid = [
    Float32.([x, y]) for (x, y) in IterTools.product(latent1_range, latent2_range)
]

# Define mask for fitness landscape
mask = (maximum(logdetG) * 0.90 .< logdetG .≤ maximum(logdetG))

## =============================================================================

println("Loading MLP model...")

# List neural network files
mlp_files = sort(Glob.glob("$(mlp_dir)/*split*.jld2"[2:end], "/"))

# Load neural network
mlp = JLD2.load("$(mlp_dir)/latent_to_phenotype.jld2")["mlp"]

## =============================================================================

# Extract data
data = JLD2.load(mlp_files[end])["data"]
# Extract neural network state
mlp_state = JLD2.load(mlp_files[end])["mlp_state"]
# Load neural network
Flux.loadmodel!(mlp, mlp_state)
# Map latent space coordinates to phenotype space. NOTE: This includes
# standardizing the input data to have mean zero and standard deviation one
# and then transforming the output data back to the original scale.
dd_mlp = DD.DimArray(
    DD.mapslices(
        slice -> StatsBase.reconstruct(
            data.transforms.x,
            mlp(StatsBase.transform(data.transforms.z, Vector(slice))),
        ),
        dd_latent,
        dims=:latent,
    ),
    (
        DD.dims(fitnotype_profiles.phenotype)[1],
        dd_latent.dims[2:end]...,
    ),
)

## =============================================================================
# Static Fig04
## =============================================================================

# Set random seed
Random.seed!(42)

println("Setting global layout...")

# Initialize figure
fig = Figure(size=(800, 600))

# ------------------------------------------------------------------------------
# Plot layout
# ------------------------------------------------------------------------------

# Add global grid layout
gl = GridLayout(fig[1, 1])

# Add grid layout for fig04A section banner
gl04A_banner = gl[1, 1] = GridLayout()
# Add grid layout for Fig04A
gl04A = gl[2, 1] = GridLayout()

# Add grid layout for fig04C section banner
gl04B_banner = gl[3, 1] = GridLayout()
# Add grid layout for Fig04C
gl04B = gl[4, 1] = GridLayout()

# Add grid layout for fig04B section banner
gl04C_banner = gl[1, 2] = GridLayout()
# Add grid layout for Fig04B
gl04C = gl[2:4, 2] = GridLayout()

# ------------------------------------------------------------------------------
# Adjust subplot proportions
# ------------------------------------------------------------------------------

# Adjust column sizes
colsize!(gl, 1, Auto(2))
colsize!(gl, 2, Auto(1))

# Adjust row sizes
rowsize!(gl, 2, Auto(4))
rowsize!(gl, 4, Auto(7))

# ------------------------------------------------------------------------------
# Add section banners
# ------------------------------------------------------------------------------

println("Adding section banners...")

# Add box for section title
Box(
    gl04A_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-15, right=-15), # Moves box to the left and right
)

# Add section title
Label(
    gl04A_banner[1, 1],
    "geometry-informed latent space with Riemannian metric",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
)

# Add box for section title
Box(
    gl04B_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-15, right=-15) # Moves box to the left and right
)

# Add section title
Label(
    gl04B_banner[1, 1],
    "comparison of ground truth and learned fitness landscapes",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
)

# Add box for section title
Box(
    gl04C_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-50), # Moves box to the left and right
)

# Add section title
Label(
    gl04C_banner[1, 1],
    "comparison of ground truth and learned\nphenotypic coordinates",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
    alignmode=Mixed(; left=-30), # Moves text to the left
    justification=:left,
)

# ------------------------------------------------------------------------------
# Fig04A
# ------------------------------------------------------------------------------

println("Plotting Fig04A...")

# Add axis
ax1 = Axis(
    gl04A[1, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    title="latent space metric",
    titlesize=14,
    xlabelsize=14,
    ylabelsize=14,
    aspect=AxisAspect(1),
    yticklabelsvisible=false,
    xticklabelsvisible=false,
)
ax2 = Axis(
    gl04A[1, 2],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    title="fitness profiles \nlatent coordinates",
    titlesize=14,
    xlabelsize=14,
    ylabelsize=14,
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
    colormap=Reverse(to_colormap(ColorSchemes.PuBu)),
)

heatmap!(
    ax2,
    latent1_range,
    latent2_range,
    logdetG,
    colormap=Reverse(to_colormap(ColorSchemes.PuBu)),
)

# Convert to Point2f
latent_points = Point2f.(
    vec(dd_latent[latent=DD.At(:latent1)]),
    vec(dd_latent[latent=DD.At(:latent2)]),
)


# Plot latent space
scatter!(
    ax2,
    latent_points,
    markersize=4,
    color=(:white, 0.3),
)

# Add colorbar
Colorbar(
    gl04A[1, 3],
    hm,
    label="√log[det(G)]",
    tellwidth=false,
    halign=:left,
)

# Adjust column gaps
colgap!(gl04A, 5)

# Adjust column sizes
colsize!(gl04A, 1, Auto(1))
colsize!(gl04A, 2, Auto(1))
colsize!(gl04A, 3, Auto(1 / 3))

# ------------------------------------------------------------------------------
# Fig04B
# ------------------------------------------------------------------------------

println("Plotting Fig04B...")

# Add grid layout for labels
gl_labels = GridLayout(gl04B[1:2, 1])
# Add fitness landscape grid layout
gl_fitness = GridLayout(gl04B[1, 2])
# Add latent space grid layout
gl_latent = GridLayout(gl04B[2, 2])

# Adjust col sizes
colsize!(gl04B, 1, Auto(0.075))
colsize!(gl04B, 2, Auto(1))

# ------------------------------------------------------------------------------

Box(
    gl_labels[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=0, right=-5, top=0, bottom=0),
)
Box(
    gl_labels[2, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=0, right=-5, top=-9, bottom=-9),
)

# Ground truth fitness landscape
Label(
    gl_labels[1, 1],
    "ground truth\nfitness landscape",
    fontsize=14,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false,
    tellheight=false,
    alignmode=Mixed(; left=0),
    rotation=π / 2,
)
# Latent space inferred fitness landscape
Label(
    gl_labels[2, 1],
    "latent space inferred\nfitness landscape",
    fontsize=14,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false,
    tellheight=false,
    alignmode=Mixed(; left=0),
    rotation=π / 2,
)

# Define environment indexes to use
env_idxs = [1, 8, 25, 36]

# Loop over fitness landscapes
for (i, env_idx) in enumerate(env_idxs)
    # Add axis
    ax_fitness = Axis(
        gl_fitness[1, i],
        title="env. $(env_idx)",
        titlesize=14,
        aspect=AxisAspect(1),
    )
    # Remove axis labels
    hidedecorations!(ax_fitness)

    # Extract fitness landscape
    fitness_landscape = fitness_landscapes[env_idx]

    # Create meshgrid for fitness landscape
    F = mh.fitness(pheno1, pheno2, fitness_landscape)

    # Plot fitness landscape
    heatmap!(ax_fitness, pheno1, pheno2, F, colormap=:algae)
    # Plot fitness landscape contour lines
    contour!(ax_fitness, pheno1, pheno2, F, color=:black, linestyle=:dash)

    # Add axis for latent fitness landscape
    ax_latent = Axis(gl_latent[1, i], aspect=AxisAspect(1))
    # Remove axis labels
    hidedecorations!(ax_latent)

    # Map latent grid to output space
    F_latent = getindex.(getindex.(rhvae.vae.decoder.(latent_grid), :μ), env_idx)

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
end # for env_idx

# Add global axis labels
Label(
    gl_fitness[end, :, Bottom()],
    "phenotype 1",
    fontsize=14,
    padding=(0, 0, 0, 0),
)
Label(
    gl_fitness[:, 1, Left()],
    "phenotype 2",
    fontsize=14,
    rotation=π / 2,
    padding=(0, 0, 0, 0),
)

# Add global axis labels
Label(
    gl_latent[end, :, Bottom()],
    "latent dimension 1",
    fontsize=14,
    padding=(0, 0, 0, 0),
)
Label(
    gl_latent[:, 1, Left()],
    "latent dimension 2",
    fontsize=14,
    rotation=π / 2,
    padding=(0, 0, 0, 0),
)

# Adjust col sizes
colgap!(gl_fitness, 5)
colgap!(gl_latent, 5)

# ------------------------------------------------------------------------------
# Fig04C
# ------------------------------------------------------------------------------

println("Plotting Fig04C...")

# Add axis for ground truth phenotypic coordinates
ax_pheno = Axis(
    gl04C[2, 1],
    title="ground truth\nphenotypic coordinates",
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    titlesize=14,
    xlabelsize=14,
    ylabelsize=14,
    aspect=AxisAspect(1),
    xticklabelsvisible=false,
    yticklabelsvisible=false
)

# Add axis for predicted phenotypic coordinates
ax_pred = Axis(
    gl04C[3, 1],
    title="learned\nphenotypic coordinates",
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    titlesize=14,
    xlabelsize=14,
    ylabelsize=14,
    aspect=AxisAspect(1),
    xticklabelsvisible=false,
    yticklabelsvisible=false
)

# Adjust row sizes
rowsize!(gl04C, 1, Auto(1 / 2))
rowsize!(gl04C, 2, Auto(1))
rowsize!(gl04C, 3, Auto(1))

# Convert to Point2f
pheno_points = Point2f.(
    vec(fitnotype_profiles.phenotype[landscape=DD.At(1), phenotype=DD.At(:x1)]),
    vec(fitnotype_profiles.phenotype[landscape=DD.At(1), phenotype=DD.At(:x2)]),
)

datashader!(
    ax_pheno,
    pheno_points,
    colormap=to_colormap(ColorSchemes.Purples),
    async=false,
    binsize=2
)

# Convert to Point2f
pred_points = Point2f.(
    vec(dd_mlp[phenotype=DD.At(:x1)]),
    vec(dd_mlp[phenotype=DD.At(:x2)]),
)

datashader!(
    ax_pred,
    pred_points,
    colormap=to_colormap(ColorSchemes.Oranges),
    async=false,
    binsize=2
)

# ------------------------------------------------------------------------------
# Add subplot labels
# ------------------------------------------------------------------------------

println("Adding subplot labels...")

# Add subplot labels
Label(
    gl04A[1, 1, TopLeft()], "(A)",
    fontsize=20,
    padding=(0, -15, 50, 0),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

Label(
    gl04B[1, 1, TopLeft()], "(B)",
    fontsize=20,
    padding=(0, -15, 25, 0),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

Label(
    gl04C[1, 1, TopLeft()], "(C)",
    fontsize=20,
    padding=(0, 50, 80, 0),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

Label(
    gl04C[1, 1, Left()], "(D)",
    fontsize=20,
    padding=(0, 0, -125, 0),
    halign=:left,
    tellwidth=false,
    tellheight=false
)

# ------------------------------------------------------------------------------
# Save figure
# ------------------------------------------------------------------------------

println("Saving figure...")

# Save figure
save("$(fig_dir)/fig04.pdf", fig)

println("Done!")

fig