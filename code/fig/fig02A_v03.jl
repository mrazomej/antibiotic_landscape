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

# Initialize figure
fig = Figure(size=(300, 500))

# Add grid layout
gl = GridLayout(fig[1, 1])

# Add grid layout for fig02A section banner
gl02A_banner = gl[1, 1] = GridLayout()
# Add grid layout for Fig02A fitness landscape
gl02A_landscape = gl[2, 1] = GridLayout()
# Add grid layout for Fig02A time series
gl02A_time = gl[3, 1] = GridLayout()

# Adjust row size
rowsize!(gl, 2, Auto(1))
rowsize!(gl, 3, Auto(3 / 4))

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
    "adaptive dynamics on single fitness function",
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
# Extract fitness trajectory
fitness_trajectory = fitnotype_profiles.fitness[
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
ax_landscape = Axis(
    gl02A_landscape[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    aspect=AxisAspect(1),
    yticklabelsvisible=false,
    xticklabelsvisible=false,
)

# Plot fitness landscape
heatmap!(ax_landscape, x, y, F, colormap=:algae)

# Plot contour lines for fitness landscape
contour!(ax_landscape, x, y, F, color=:black, linestyle=:dash)

# Plot contour lines for genetic density
contour!(ax_landscape, x, y, G, color=:white, linestyle=:solid)

# ------------------------------------------------------------------------------

# Add axis for time series
ax_time = Axis(
    gl02A_time[1, 1],
    xlabel="time (a.u.)",
    ylabel="fitness (a.u.)",
    aspect=AxisAspect(4 / 3),
)

# ------------------------------------------------------------------------------

# Initialize counter
counter = 1
# Loop over lineages
for lin in DD.dims(phenotype_trajectory, :lineage)
    # Loop over replicates
    for rep in DD.dims(phenotype_trajectory, :replicate)
        # Plot trajectory
        lines!(
            ax_landscape,
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
        # Plot fitness trajectory
        lines!(
            ax_time,
            vec(fitness_trajectory[
                lineage=lin,
                replicate=rep,
            ].data),
            color=ColorSchemes.glasbey_hv_n256[counter],
        )
        # Increment counter
        global counter += 1
    end # for rep
end # for lin

# Set axis limits
xlims!(ax_landscape, phenotype_lims.x...)
ylims!(ax_landscape, phenotype_lims.y...)

# Save figure
save("$(fig_dir)/fig02A.pdf", fig)

fig

## =============================================================================

println("Creating animated evolutionary trajectories...")

# Create figure for animation with same dimensions as static plot
fig = Figure(size=(300, 500))

# Add grid layout
gl = GridLayout(fig[1, 1])

# Add grid layouts matching static plot
gl02A_banner = gl[1, 1] = GridLayout()
gl02A_landscape = gl[2, 1] = GridLayout()
gl02A_time = gl[3, 1] = GridLayout()

# Adjust row size to match static plot
rowsize!(gl, 2, Auto(1))
rowsize!(gl, 3, Auto(3 / 4))

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
    "adaptive dynamics on single fitness function",
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
# Extract fitness trajectory
fitness_trajectory = fitnotype_profiles.fitness[
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

# Add axes for both panels
ax_landscape = Axis(
    gl02A_landscape[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    aspect=AxisAspect(1),
    yticklabelsvisible=false,
    xticklabelsvisible=false,
)

ax_time = Axis(
    gl02A_time[1, 1],
    xlabel="time (a.u.)",
    ylabel="fitness (a.u.)",
    aspect=AxisAspect(4 / 3),
)

# Plot static elements
heatmap!(ax_landscape, x, y, F, colormap=:algae)
contour!(ax_landscape, x, y, F, color=:black, linestyle=:dash)
contour!(ax_landscape, x, y, G, color=:white, linestyle=:solid)

# Set axis limits for fitness landscape
xlims!(ax_landscape, phenotype_lims.x...)
ylims!(ax_landscape, phenotype_lims.y...)
# Set axis limits for time series
xlims!(ax_time, 0, length(DD.dims(fitnotype_profiles, :time)))
ylims!(
    ax_time,
    minimum(fitness_trajectory) - 0.1,
    maximum(fitness_trajectory) + 0.1
)

# Create dictionaries to store both trajectory and fitness data
trajectory = Dict()
fitness_data = Dict()
counter = 1

# Store both phenotype and fitness data
for lin in DD.dims(phenotype_trajectory, :lineage)
    for rep in DD.dims(phenotype_trajectory, :replicate)
        # Store phenotype trajectory
        trajectory[counter] = phenotype_trajectory[
            lineage=lin,
            replicate=rep,
        ].data
        # Store fitness trajectory
        fitness_data[counter] = vec(fitness_trajectory[
            lineage=lin,
            replicate=rep,
        ].data)
        # Increment counter
        global counter += 1
    end
end

# Create dictionaries of observables for both plots
points_dict = Dict(
    i => Observable(Point2f[trajectory[i][:, 1]])
    for i in 1:length(trajectory)
)

fitness_points_dict = Dict(
    i => Observable(Point2f[(1, fitness_data[i][1])])
    for i in 1:length(trajectory)
)

# Plot initial lines for each trajectory in both panels
for (i, points) in points_dict
    lines!(
        ax_landscape,
        points,
        color=ColorSchemes.glasbey_hv_n256[i],
        linewidth=1.5
    )
    lines!(
        ax_time,
        fitness_points_dict[i],
        color=ColorSchemes.glasbey_hv_n256[i],
        linewidth=1.5
    )
end

# Record animation updating both panels
record(
    fig, "$(fig_dir)/fig02A.mp4",
    1:size(trajectory[1], 2);
    framerate=60
) do frame
    for i in 1:length(trajectory)
        # Update phenotype trajectory
        new_point = Point2f(trajectory[i][:, frame])
        points_dict[i][] = push!(points_dict[i][], new_point)

        # Update fitness trajectory
        new_fitness_point = Point2f([frame, fitness_data[i][frame]])
        fitness_points_dict[i][] = push!(fitness_points_dict[i][], new_fitness_point)
    end
end