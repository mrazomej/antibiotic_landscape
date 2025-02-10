## =============================================================================

println("Importing packages...")

# Import project package
import Antibiotic
import Antibiotic.mh as mh

# Import packages for storing results
import DimensionalData as DD
import StructArrays as SA

# Import JLD2 for saving results
import JLD2

# Import basic math libraries
import StatsBase
import LinearAlgebra
import Random
import Distributions
import Distances
# Load CairoMakie for plotting
using GLMakie
import ColorSchemes

# Activate backend
GLMakie.activate!()
# Set PBoC Plotting style
Antibiotic.viz.theme_makie!()

Random.seed!(42)

## =============================================================================

println("Defining directories...")

# Locate current directory
path_dir = pwd()

# Find the path perfix where input data is stored
out_prefix = replace(
    match(r"processing/(.*)", path_dir).match,
    "processing" => "",
)

# Define simulation directory
sim_dir = "$(git_root())/output$(out_prefix)/sim_evo"

# Define figure directory
fig_dir = "$(git_root())/fig$(out_prefix)/sim_evo"

# Generate figure directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating figure directory...")
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

println("Evaluating example fitness landscape...")

# Define index of fitness landscape to plot
evo = 24
# Select fitness landscape
fit_lan = fitness_landscapes[evo]

# Define ranges of phenotypes to evaluate
x = y = z = range(-6, 6, length=50)
coords = (x, y, z)

# Evaluate fitness landscape
F = mh.fitness(coords, fit_lan)

# Extract dimensions
dims = DD.dims(F)


## =============================================================================

# Define pairs of coordinates
panel_idx = [
    idx for idx in CartesianIndices((length(coords), length(coords)))
    if idx[1] < idx[2]
]

# Define panel name dictionary
panel_names = Dict(
    1 => "x",
    2 => "y",
    3 => "z",
)

## =============================================================================

println("Plotting example fitness landscape...")

# Initialize figure
fig = Figure(size=(500, 500))

# Define layout
gl = fig[1, 1] = GridLayout()

# Add 3D axis
ax = Axis3(
    gl[1, 1],
    aspect=(1, 1, 1),
    xgridvisible=false,
    ygridvisible=false,
    zgridvisible=false,
)

# Hide decorations
hidedecorations!(ax)

# Loop over pairs of coordinates
for (j, p_idx) in enumerate(panel_idx)
    # Extract coordinates
    x, y = p_idx.I
    # Extract z coordinate as the missing coordinate
    z = setdiff([1, 2, 3], [x, y])[1]
    # Marginalize over the other dimension
    F_marg = sum(
        F, dims=dims[[d for d in 1:length(dims) if d != x && d != y]]
    )
    # Drop dimensions that were marginalized over
    # NOTE: The dims argument must be a tuple of the dimensions to drop.
    F_marg = dropdims(
        F_marg.data,
        dims=tuple(findall(size(F_marg.data) .== 1)...)
    )
    # Extract panel name
    panel = Symbol("$(panel_names[x])$(panel_names[y])")
    # Extract panel position
    panel_pos = Dict(
        :xy => minimum(coords[x]),
        :xz => maximum(coords[y]),
        :yz => maximum(coords[z]),
    )[panel]

    # Plot fitness landscape
    contourf!(
        ax,
        coords[x],
        coords[y],
        F_marg,
        colormap=:algae,
        transformation=(panel, panel_pos),
    )
end

# Set plot limits
xlims!(ax, (minimum(coords[1]) - 0.05, maximum(coords[1]) + 0.05))
ylims!(ax, (minimum(coords[2]) - 0.05, maximum(coords[2]) + 0.05))
zlims!(ax, (minimum(coords[3]) - 0.05, maximum(coords[3]) + 0.05))

# Display figure
fig

## =============================================================================

println("Plotting example fitness landscape with trajectories...")

# Extract phenotypic trajectories evolved on the fitness landscape
evo_traj = fitnotype_profiles.phenotype[evo=DD.At(evo), landscape=DD.At(evo)]

# Initialize figure
fig = Figure(size=(500, 500))

# Define layout
gl = fig[1, 1] = GridLayout()

# Add 3D axis
ax = Axis3(
    gl[1, 1],
    aspect=(1, 1, 1),
    xgridvisible=false,
    ygridvisible=false,
    zgridvisible=false,
)

# Hide decorations
hidedecorations!(ax)

# Initialize counter
counter = 1
# Plot fitness trajectories
for lin in DD.dims(evo_traj, :lineage)
    for rep in DD.dims(evo_traj, :replicate)
        # Extract fitness trajectory
        local x, y, z = eachrow(evo_traj[
            lineage=lin,
            replicate=rep
        ].data)
        # Plot fitness trajectories
        lines!(
            ax,
            x,
            y,
            z,
            linewidth=2,
            color=ColorSchemes.glasbey_hv_n256[counter],
        )
        # Increment counter
        global counter += 1
    end
end

# Loop over pairs of coordinates
for (j, p_idx) in enumerate(panel_idx)
    # Extract coordinates
    x, y = p_idx.I
    # Extract z coordinate as the missing coordinate
    z = setdiff([1, 2, 3], [x, y])[1]
    # Marginalize over the other dimension
    F_marg = sum(
        F, dims=dims[[d for d in 1:length(dims) if d != x && d != y]]
    )
    # Drop dimensions that were marginalized over
    # NOTE: The dims argument must be a tuple of the dimensions to drop.
    F_marg = dropdims(
        F_marg.data,
        dims=tuple(findall(size(F_marg.data) .== 1)...)
    )
    # Extract panel name
    panel = Symbol("$(panel_names[x])$(panel_names[y])")
    # Extract panel position
    panel_pos = Dict(
        :xy => minimum(coords[x]),
        :xz => maximum(coords[y]),
        :yz => maximum(coords[z]),
    )[panel]

    # Plot fitness landscape
    contourf!(
        ax,
        coords[x],
        coords[y],
        F_marg,
        colormap=:algae,
        transformation=(panel, panel_pos),
    )
end

# Set plot limits
xlims!(ax, (minimum(coords[1]) - 0.05, maximum(coords[1]) + 0.05))
ylims!(ax, (minimum(coords[2]) - 0.05, maximum(coords[2]) + 0.05))
zlims!(ax, (minimum(coords[3]) - 0.05, maximum(coords[3]) + 0.05))

# Display figure
fig

