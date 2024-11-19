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

# Define number of rows and columns
n_rows = 3
n_cols = 3

# Initialize figure
fig = Figure(size=(200 * n_cols, 200 * n_rows))

# Add grid layout
gl = fig[1, 1] = GridLayout()

# Define limits of phenotype space
phenotype_lims = (
    x=(-5, 5),
    y=(-5, 5),
)

# Define range of phenotypes to evaluate
x = range(phenotype_lims.x..., length=100)
y = range(phenotype_lims.y..., length=100)

# Create meshgrid for genetic density
G = mh.genetic_density(x, y, genetic_density)

# Loop over plots
for i in 1:(n_rows*n_cols)
    # Define row and column
    row = (i - 1) ÷ n_cols + 1
    col = (i - 1) % n_cols + 1

    # Add axis
    ax = Axis(gl[row, col])
    # Remove axis labels
    hidedecorations!(ax)

    # Define landscape index
    landscape_idx = i + 1

    # Extract fitness landscape
    fitness_landscape = fitness_landscapes[landscape_idx]

    # Create meshgrid for fitness landscape
    F = mh.fitness(x, y, fitness_landscape)


    # Plot fitness landscape
    heatmap!(ax, x, y, F, colormap=:viridis)
    # Plot fitness landscape contour lines
    contour!(ax, x, y, F, color=:black, linestyle=:dash)

    # Plot genetic density contour lines
    contour!(ax, x, y, G, color=:white, linestyle=:solid)

    # Extract data
    phenotype_trajectory = fitnotype_profiles.phenotype[
        evo=landscape_idx, landscape=landscape_idx
    ]

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
end # for i

# Adjust subplot spacing
colgap!(gl, 3)
rowgap!(gl, 3)

# Add global axis labels
Label(
    gl[end, :, Bottom()],
    "phenotype 1",
    fontsize=16,
    padding=(0, 0, -15, 5),
)
Label(
    gl[:, 1, Left()],
    "phenotype 2",
    fontsize=16,
    rotation=π / 2,
    padding=(-15, 5, 0, 0),
)

# Save figure
save("$(fig_dir)/fig02B_v01.pdf", fig)

fig