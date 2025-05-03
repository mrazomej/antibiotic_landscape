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
sim_dir = "$(git_root())/output/metropolis-hastings_sim/v05/sim_evo"

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

println("Plotting alternative landscapes adaptive dynamics...")

# Define number of rows and columns
n_rows = 3
n_cols = 3

# Initialize figure
fig = Figure(size=(200 * n_cols, 200 * n_rows))

# Add grid layout
gl = fig[1, 1] = GridLayout()

# Add grid layout for fig02B section banner
gl02B_banner = gl[1, 1] = GridLayout()
# Add grid layout for Fig02B
gl02B = gl[2, 1] = GridLayout()

# Add box for section title
Box(
    gl02B_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-30, right=-20)
)

Label(
    gl02B_banner[1, 1],
    "alternative landscapes adaptive dynamics",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false,
    alignmode=Mixed(; left=-10)
)

# ------------------------------------------------------------------------------

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
    ax = Axis(gl02B[row, col])
    # Remove axis labels
    hidedecorations!(ax)

    # Define landscape index
    landscape_idx = i + 1

    # Extract fitness landscape
    fitness_landscape = fitness_landscapes[landscape_idx]

    # Create meshgrid for fitness landscape
    F = mh.fitness(x, y, fitness_landscape)


    # Plot fitness landscape
    heatmap!(ax, x, y, F, colormap=:algae)
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
colgap!(gl02B, 3)
rowgap!(gl02B, 3)

# Add global axis labels
Label(
    gl02B[end, :, Bottom()],
    "phenotype 1",
    fontsize=16,
    padding=(0, 0, -15, 5),
)
Label(
    gl02B[:, 1, Left()],
    "phenotype 2",
    fontsize=16,
    rotation=π / 2,
    padding=(-15, 5, 0, 0),
)

# Save figure
save("$(fig_dir)/fig02B.pdf", fig)

fig

## =============================================================================

println("Creating animated alternative landscapes adaptive dynamics...")

println("Creating animated 3x3 grid of landscapes...")

# Initialize figure with same dimensions as static plot
fig = Figure(size=(200 * n_cols, 200 * n_rows))

# Add grid layouts
gl = fig[1, 1] = GridLayout()
gl02B_banner = gl[1, 1] = GridLayout()
gl02B = gl[2, 1] = GridLayout()

# Add banner box and label (same as static plot)
Box(
    gl02B_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-30, right=-20)
)

Label(
    gl02B_banner[1, 1],
    "alternative landscapes adaptive dynamics",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false,
    alignmode=Mixed(; left=-10)
)

# Create dictionary to store axes and trajectories
axes_dict = Dict()
trajectories_dict = Dict()
points_dict = Dict()

# Initialize all axes and static elements
for i in 1:(n_rows*n_cols)
    row = (i - 1) ÷ n_cols + 1
    col = (i - 1) % n_cols + 1
    landscape_idx = i + 1

    # Create axis
    ax = Axis(gl02B[row, col])
    hidedecorations!(ax)
    axes_dict[i] = ax

    # Plot static elements
    fitness_landscape = fitness_landscapes[landscape_idx]
    F = mh.fitness(x, y, fitness_landscape)

    heatmap!(ax, x, y, F, colormap=:algae)
    contour!(ax, x, y, F, color=:black, linestyle=:dash)
    contour!(ax, x, y, G, color=:white, linestyle=:solid)

    xlims!(ax, phenotype_lims.x...)
    ylims!(ax, phenotype_lims.y...)

    # Extract and store trajectory data
    phenotype_trajectory = fitnotype_profiles.phenotype[
        evo=landscape_idx, landscape=landscape_idx
    ]

    trajectories_dict[i] = Dict()
    points_dict[i] = Dict()
    counter = 1

    for lin in DD.dims(phenotype_trajectory, :lineage)
        for rep in DD.dims(phenotype_trajectory, :replicate)
            trajectories_dict[i][counter] = [
                Point2f(
                    phenotype_trajectory[phenotype=DD.At(:x1), lineage=lin, replicate=rep].data[j],
                    phenotype_trajectory[phenotype=DD.At(:x2), lineage=lin, replicate=rep].data[j]
                ) for j in 1:length(phenotype_trajectory[phenotype=DD.At(:x1), lineage=lin, replicate=rep].data)
            ]

            # Create observable for this trajectory
            points_dict[i][counter] = Observable([trajectories_dict[i][counter][1]])

            # Plot initial point
            lines!(ax, points_dict[i][counter], color=ColorSchemes.glasbey_hv_n256[counter], linewidth=1.5)

            counter += 1
        end
    end
end

# Add global axis labels
Label(
    gl02B[end, :, Bottom()],
    "phenotype 1",
    fontsize=16,
    padding=(0, 0, -15, 5),
)
Label(
    gl02B[:, 1, Left()],
    "phenotype 2",
    fontsize=16,
    rotation=π / 2,
    padding=(-15, 5, 0, 0),
)

# Adjust subplot spacing
colgap!(gl02B, 3)
rowgap!(gl02B, 3)

# Create animation
record(
    fig, "$(fig_dir)/fig02B.mp4",
    1:length(first(values(first(values(trajectories_dict)))));
    framerate=60
) do frame
    # Update all trajectories in all plots
    for i in 1:(n_rows*n_cols)
        for (traj_idx, _) in points_dict[i]
            points_dict[i][traj_idx][] = trajectories_dict[i][traj_idx][1:frame]
        end
    end
end