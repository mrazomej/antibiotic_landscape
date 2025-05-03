## =============================================================================

println("Importing packages...")

# Import project package
import Antibiotic
import Antibiotic.mh as mh

# Load CairoMakie for plotting
using CairoMakie
import ColorSchemes

# Import packages for storing results
import DimensionalData as DD

# Import JLD2 for loading results
import JLD2

# Import Distributions
import Distributions

# Import Random
import Random

# Activate backend
CairoMakie.activate!()

# Set PBoC Plotting style
Antibiotic.viz.theme_makie!()

# Set random seed
Random.seed!(42)

## =============================================================================

println("Defining directories...")

# Define simulation directory
sim_dir = "$(git_root())/output/metropolis-hastings_sim/v06/sim_evo"

# Define output directory
fig_dir = "$(git_root())/fig/main"

# Generate output directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating output directory...")
    mkpath(fig_dir)
end

## =============================================================================

println("Loading simulation results...")

# Load fitnotype profiles
fitnotype_profiles = JLD2.load("$(sim_dir)/sim_evo.jld2")["fitnotype_profiles"]
# Load fitness landscapes
fitness_landscapes = JLD2.load("$(sim_dir)/sim_evo.jld2")["fitness_landscapes"]
# Load mutational landscape
genetic_density = JLD2.load(
    "$(sim_dir)/sim_evo.jld2"
)["genetic_density"]

## =============================================================================

println("Plotting example evolutionary trajectories...")

# Define landscape index
landscape_idx = 1

# Extract data
phenotype_trajectory = fitnotype_profiles.phenotype[
    evo=landscape_idx, landscape=landscape_idx
]

# Define limits of phenotype space
phenotype_lims = (
    x=(-5, 5),
    y=(-5, 5),
)

# Define range of phenotypes to evaluate
x = range(phenotype_lims.x..., length=100)
y = range(phenotype_lims.y..., length=100)

# Create meshgrid
F = mh.fitness(x, y, fitness_landscapes[landscape_idx])
G = mh.genetic_density(x, y, genetic_density)

# Initialize figure
fig = Figure(size=(300, 300))

# Add axis for fitness landscape
ax = Axis(
    fig[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    aspect=AxisAspect(1),
)

# Plot fitness landscape
heatmap!(ax, x, y, F, colormap=:viridis)

# Plot contour lines for fitness landscape
contour!(ax, x, y, F, color=:black, linestyle=:dash)

# Plot contour lines for genetic density
contour!(ax, x, y, G, color=:white, linestyle=:solid)

# Initialize counter
counter = 1
# Loop over lineages
for lin in DD.dims(phenotype_trajectory, :lineage)
    # Loop over replicates
    for rep in DD.dims(phenotype_trajectory, :replicate)
        # Plot trajectory
        lines!(
            ax,
            phenotype_trajectory[
                phenotype=DD.At(:x1),
                lineage=lin,
                replicate=rep,
            ].data,
            phenotype_trajectory[
                phenotype=DD.At(:x2),
                lineage=lin,
                replicate=rep,
            ].data,
            color=ColorSchemes.glasbey_hv_n256[counter],
            linewidth=1.5
        )
        # Increment counter
        counter += 1
    end # for rep
end # for lin

# Set axis limits
xlims!(ax, phenotype_lims.x...)
ylims!(ax, phenotype_lims.y...)

fig
