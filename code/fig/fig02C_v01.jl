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

println("Plotting fitness profiles over time...")

# Define rows and columns
rows = 2
cols = 4

# Initialize figure
fig = Figure(size=(200 * cols, 200 * rows))

# Add global grid layout
gl = GridLayout(fig[1, 1])

# Loop through rows and columns
for i in 1:(rows*cols)
    # Define row and column
    row = (i - 1) ÷ cols + 1
    col = (i - 1) % cols + 1

    # Define axis
    ax = Axis(fig[row, col])

    # Add axis title
    if (col == 1) & (row == 1)
        ax.title = "env. $(i) (selection)"
    else
        ax.title = "env. $(i)"
    end # if

    # Initialize counter
    counter = 1

    # Loop through replicates
    for rep in DD.dims(fitnotype_profiles, :replicate)
        # Loop through lineage
        for lin in DD.dims(fitnotype_profiles, :lineage)
            # Extract fitness profile
            fitness_profile = fitnotype_profiles.fitness[
                evo=1,
                landscape=i,
                replicate=rep,
                lineage=lin
            ]

            # Plot fitness profile
            lines!(
                ax,
                vec(fitness_profile),
                color=ColorSchemes.glasbey_hv_n256[counter],
            )

            # Increment counter
            counter += 1
        end # for lineage
    end # for rep
end # for i

# Add global axis labels
Label(
    fig[end, :, Bottom()],
    "time (a.u.)",
    fontsize=16,
    padding=(0, 0, -15, 20),
)
Label(
    fig[:, 1, Left()],
    "fitness (a.u.)",
    fontsize=16,
    rotation=π / 2,
    padding=(-15, 20, 0, 0),
)

# Save figure
save("$(fig_dir)/fig02C_v01.pdf", fig)

fig