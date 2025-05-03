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

## =============================================================================

# Define limits of phenotype space
phenotype_lims = (
    x=(-5, 5),
    y=(-5, 5),
)

# Define range of phenotypes to evaluate
pheno1 = range(phenotype_lims.x..., length=100)
pheno2 = range(phenotype_lims.y..., length=100)

# Create meshgrid for genetic density
G = mh.genetic_density(pheno1, pheno2, genetic_density)

# Convert product to array of vectors
latent_grid = [
    Float32.([x, y]) for (x, y) in IterTools.product(latent1_range, latent2_range)
]

# Define mask for fitness landscape
mask = (maximum(logdetG) * 0.90 .< logdetG .≤ maximum(logdetG))

## =============================================================================

println("Plotting fitness landscapes...")

# Initialize figure
fig = Figure(size=(160 * 4, 200 * 2))

# Add global grid layout
gl = GridLayout(fig[1, 1])

# Add grid layout for title banner
gl_banner = GridLayout(gl[1, 1:2])
# Add grid layout for labels
gl_labels = GridLayout(gl[2:3, 1])
# Add fitness landscape grid layout
gl_fitness = GridLayout(gl[2, 2])
# Add latent space grid layout
gl_latent = GridLayout(gl[3, 2])

# ------------------------------------------------------------------------------
# Add labels
# ------------------------------------------------------------------------------

# Add box for section title
Box(
    gl_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-300, right=-200) # Moves box to the left and right
)
Box(
    gl_labels[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-5, right=-5) # Moves box to the left and right
)
Box(
    gl_labels[2, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-5, right=-5) # Moves box to the left and right
)
# Title
Label(
    gl_banner[1, 1],
    "comparison of ground truth and latent space inferred fitness landscapes",
    padding=(0, 0, 0, 0),
    halign=:left,
    alignmode=Mixed(; left=-150),
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

# Adjust col sizes
colsize!(gl, 1, Auto(0.05))
colsize!(gl, 2, Auto(1))

# ------------------------------------------------------------------------------
# Plot fitness landscapes
# ------------------------------------------------------------------------------

# Define environment indexes to use
env_idxs = [1, 8, 25, 36]

# Loop over fitness landscapes
for (i, env_idx) in enumerate(env_idxs)
    # --------------------------------------------------------------------------
    # Plot ground truth fitness landscape
    # --------------------------------------------------------------------------

    # Add axis
    ax_fitness = Axis(
        gl_fitness[1, i],
        title="env. $(env_idx)",
        aspect=AxisAspect(1),
    )
    # Remove axis labels
    hidedecorations!(ax_fitness)
    # Set axis limits
    xlims!(ax_fitness, phenotype_lims.x...)
    ylims!(ax_fitness, phenotype_lims.y...)

    fitness_landscape = fitness_landscapes[env_idx]

    # Create meshgrid for fitness landscape
    F = mh.fitness(pheno1, pheno2, fitness_landscape)

    # Plot fitness landscape
    heatmap!(ax_fitness, pheno1, pheno2, F, colormap=:algae)
    # Plot fitness landscape contour lines
    contour!(ax_fitness, pheno1, pheno2, F, color=:black, linestyle=:dash)

    # --------------------------------------------------------------------------
    # Plot example trajectories
    # --------------------------------------------------------------------------

    # Extract data for environment
    pheno_data = fitnotype_profiles.phenotype[
        landscape=1, evo=env_idx, replicate=1
    ]

    # Initialize counter
    counter = 1
    # Loop over lineages
    for lin in DD.dims(pheno_data, :lineage)
        # Extract trajectory
        lines!(
            ax_fitness,
            pheno_data[lineage=lin, phenotype=DD.At(:x1)].data,
            pheno_data[lineage=lin, phenotype=DD.At(:x2)].data,
            color=ColorSchemes.glasbey_hv_n256[counter],
            linewidth=1.5
        )
        # Increment counter
        counter += 1
    end # for lin

    # --------------------------------------------------------------------------
    # Plot latent fitness landscape
    # --------------------------------------------------------------------------

    # Add axis for latent fitness landscape
    ax_latent = Axis(gl_latent[1, i], aspect=AxisAspect(1))
    # Remove axis labels
    hidedecorations!(ax_latent)

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
        lines!(
            ax_latent,
            latent_data[lineage=lin, latent=DD.At(:latent1)].data,
            latent_data[lineage=lin, latent=DD.At(:latent2)].data,
            color=ColorSchemes.glasbey_hv_n256[counter],
            linewidth=1.5
        )
        # Increment counter
        counter += 1
    end # for lin
end # for env_idx

# Add global axis labels
Label(
    gl_fitness[end, :, Bottom()],
    "phenotype 1",
    fontsize=16,
    padding=(0, 0, 0, 5),
)
Label(
    gl_fitness[:, 1, Left()],
    "phenotype 2",
    fontsize=16,
    rotation=π / 2,
    padding=(0, -10, 0, 0),
)

# Add global axis labels
Label(
    gl_latent[end, :, Bottom()],
    "latent dimension 1",
    fontsize=16,
    padding=(0, 0, 0, 5),
)
Label(
    gl_latent[:, 1, Left()],
    "latent dimension 2",
    fontsize=16,
    rotation=π / 2,
    padding=(0, -10, 0, 0),
)

# Adjust col sizes
colgap!(gl_fitness, -10)
colgap!(gl_latent, -10)

# Save figure
save("$(fig_dir)/fig04B.pdf", fig)

fig

## =============================================================================

println("Creating animated fitness landscapes...")

# Initialize figure
fig = Figure(size=(160 * 4, 200 * 2))

# Add global grid layout
gl = GridLayout(fig[1, 1])

# Add grid layout for title banner
gl_banner = GridLayout(gl[1, 1:2])
# Add grid layout for labels
gl_labels = GridLayout(gl[2:3, 1])
# Add fitness landscape grid layout
gl_fitness = GridLayout(gl[2, 2])
# Add latent space grid layout
gl_latent = GridLayout(gl[3, 2])

# ------------------------------------------------------------------------------
# Add labels
# ------------------------------------------------------------------------------

# Add box for section title
Box(
    gl_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-300, right=-200) # Moves box to the left and right
)
Box(
    gl_labels[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-5, right=-5) # Moves box to the left and right
)
Box(
    gl_labels[2, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-5, right=-5) # Moves box to the left and right
)
# Title
Label(
    gl_banner[1, 1],
    "comparison of ground truth and latent space inferred fitness landscapes",
    padding=(0, 0, 0, 0),
    halign=:left,
    alignmode=Mixed(; left=-150),
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

# Adjust col sizes
colsize!(gl, 1, Auto(0.05))
colsize!(gl, 2, Auto(1))

# ------------------------------------------------------------------------------
# Plot fitness landscapes
# ------------------------------------------------------------------------------

# Define environment indexes to use
env_idxs = [1, 8, 25, 36]

# Create dictionaries to store trajectories and points
fitness_trajectories = Dict()
fitness_points = Dict()
latent_trajectories = Dict()
latent_points = Dict()

# Loop over fitness landscapes
for (i, env_idx) in enumerate(env_idxs)
    # --------------------------------------------------------------------------
    # Plot ground truth fitness landscape
    # --------------------------------------------------------------------------

    # Add axis
    ax_fitness = Axis(
        gl_fitness[1, i],
        title="env. $(env_idx)",
        aspect=AxisAspect(1),
    )
    # Remove axis labels
    hidedecorations!(ax_fitness)
    # Set axis limits
    xlims!(ax_fitness, phenotype_lims.x...)
    ylims!(ax_fitness, phenotype_lims.y...)

    # Plot static elements (heatmap and contours)
    fitness_landscape = fitness_landscapes[env_idx]
    F = mh.fitness(pheno1, pheno2, fitness_landscape)
    heatmap!(ax_fitness, pheno1, pheno2, F, colormap=:algae)
    contour!(ax_fitness, pheno1, pheno2, F, color=:black, linestyle=:dash)

    # Store trajectory data for animation
    pheno_data = fitnotype_profiles.phenotype[
        landscape=1, evo=env_idx, replicate=1
    ]

    fitness_trajectories[i] = Dict()
    fitness_points[i] = Dict()
    counter = 1

    for lin in DD.dims(pheno_data, :lineage)
        # Create trajectory points
        fitness_trajectories[i][counter] = [
            Point2f(
                pheno_data[lineage=lin, phenotype=DD.At(:x1)].data[j],
                pheno_data[lineage=lin, phenotype=DD.At(:x2)].data[j]
            ) for j in 1:length(pheno_data[lineage=lin, phenotype=DD.At(:x1)].data)
        ]

        # Create observable for animation
        fitness_points[i][counter] = Observable([fitness_trajectories[i][counter][1]])

        # Plot initial point
        lines!(
            ax_fitness,
            fitness_points[i][counter],
            color=ColorSchemes.glasbey_hv_n256[counter],
            linewidth=1.5
        )
        counter += 1
    end

    # --------------------------------------------------------------------------
    # Plot latent fitness landscape
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Plot latent fitness landscape
    # --------------------------------------------------------------------------

    # Add axis for latent fitness landscape
    ax_latent = Axis(gl_latent[1, i], aspect=AxisAspect(1))
    # Remove axis labels
    hidedecorations!(ax_latent)

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

    # Store latent trajectory data
    latent_data = dd_latent[landscape=1, evo=env_idx, replicate=1]

    latent_trajectories[i] = Dict()
    latent_points[i] = Dict()
    counter = 1

    for lin in DD.dims(latent_data, :lineage)
        latent_trajectories[i][counter] = [
            Point2f(
                latent_data[lineage=lin, latent=DD.At(:latent1)].data[j],
                latent_data[lineage=lin, latent=DD.At(:latent2)].data[j]
            ) for j in 1:length(latent_data[lineage=lin, latent=DD.At(:latent1)].data)
        ]

        latent_points[i][counter] = Observable([latent_trajectories[i][counter][1]])

        lines!(
            ax_latent,
            latent_points[i][counter],
            color=ColorSchemes.glasbey_hv_n256[counter],
            linewidth=1.5
        )
        counter += 1
    end
end

# Add global axis labels
Label(
    gl_fitness[end, :, Bottom()],
    "phenotype 1",
    fontsize=16,
    padding=(0, 0, 0, 5),
)
Label(
    gl_fitness[:, 1, Left()],
    "phenotype 2",
    fontsize=16,
    rotation=π / 2,
    padding=(0, -10, 0, 0),
)

# Add global axis labels
Label(
    gl_latent[end, :, Bottom()],
    "latent dimension 1",
    fontsize=16,
    padding=(0, 0, 0, 5),
)
Label(
    gl_latent[:, 1, Left()],
    "latent dimension 2",
    fontsize=16,
    rotation=π / 2,
    padding=(0, -10, 0, 0),
)

# Adjust col sizes
colgap!(gl_fitness, -10)
colgap!(gl_latent, -10)

# Create animation
record(
    fig, "$(fig_dir)/fig04B.mp4",
    1:length(first(values(first(values(fitness_trajectories)))));
    framerate=10
) do frame
    # Update all trajectories in all plots
    for i in 1:length(env_idxs)
        for (traj_idx, _) in fitness_points[i]
            fitness_points[i][traj_idx][] = fitness_trajectories[i][traj_idx][1:frame]
            latent_points[i][traj_idx][] = latent_trajectories[i][traj_idx][1:frame]
        end
    end
end

fig