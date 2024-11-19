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

# Initialize figure
fig = Figure(size=(300, 300))

# Add grid layout
gl = GridLayout(fig[1, 1])

# Add grid layout for fig02A section banner
gl02A_banner = gl[1, 1] = GridLayout()
# Add grid layout for Fig02A
gl02A = gl[2, 1] = GridLayout()

# ------------------------------------------------------------------------------

# Add box for section title
Box(
    gl02A_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-50) # Moves box to the left and right
)

# Add section title
Label(
    gl02A_banner[1, 1],
    "adaptive dynamics on fitness landscape",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
    alignmode=Mixed(; left=-30) # Moves text to the left
)

# ------------------------------------------------------------------------------

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

# Add axis for fitness landscape
ax = Axis(
    gl02A[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    aspect=AxisAspect(1),
    yticklabelsvisible=false,
    xticklabelsvisible=false,
)

# Plot fitness landscape
heatmap!(ax, x, y, F, colormap=:algae)

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
        global counter += 1
    end # for rep
end # for lin

# Set axis limits
xlims!(ax, phenotype_lims.x...)
ylims!(ax, phenotype_lims.y...)

# Save figure
save("$(fig_dir)/fig02A.pdf", fig)

fig

## =============================================================================

println("Creating animated evolutionary trajectories...")

# Create figure for animation
fig = Figure(size=(300, 300))

# Add grid layout
gl = GridLayout(fig[1, 1])

# Add grid layout for fig02A section banner
gl02A_banner = gl[1, 1] = GridLayout()
# Add grid layout for Fig02A
gl02A = gl[2, 1] = GridLayout()

# ------------------------------------------------------------------------------

# Add box for section title
Box(
    gl02A_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-50) # Moves box to the left and right
)

# Add section title
Label(
    gl02A_banner[1, 1],
    "adaptive dynamics on fitness landscape",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
    alignmode=Mixed(; left=-30) # Moves text to the left
)

# ------------------------------------------------------------------------------

# Add axis for fitness landscape
ax = Axis(
    gl02A[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    # title=@lift("frame $(Int($current_frame))"),
    aspect=AxisAspect(1),
    yticklabelsvisible=false,
    xticklabelsvisible=false,
)

# Plot fitness landscape
heatmap!(ax, x, y, F, colormap=:algae)
contour!(ax, x, y, F, color=:black, linestyle=:dash)
contour!(ax, x, y, G, color=:white, linestyle=:solid)

# Set axis limits
xlims!(ax, phenotype_lims.x...)
ylims!(ax, phenotype_lims.y...)

# Create dictionary to store trajectory data for each lineage/replicate
trajectory = Dict()
# Initialize counter
counter = 1
# Loop over lineages
for lin in DD.dims(phenotype_trajectory, :lineage)
    # Loop over replicates
    for rep in DD.dims(phenotype_trajectory, :replicate)
        # Extract data
        trajectory[counter] = phenotype_trajectory[
            lineage=lin,
            replicate=rep,
        ].data
        # Increment counter
        global counter += 1
    end
end

# Create dictionary of observables for each trajectory
points_dict = Dict(
    i => Observable(Point2f[trajectory[i][:, 1]])
    for i in 1:length(trajectory)
)

# Plot lines for each trajectory
for (i, points) in points_dict
    lines!(ax, points, color=ColorSchemes.glasbey_hv_n256[i], linewidth=1.5)
end

record(
    fig, "$(fig_dir)/fig02A.mp4",
    1:size(trajectory[1], 2);
    framerate=60
) do frame
    # Update each trajectory
    for i in 1:length(trajectory)
        new_point = Point2f(trajectory[i][:, frame])
        points_dict[i][] = push!(points_dict[i][], new_point)
    end
end