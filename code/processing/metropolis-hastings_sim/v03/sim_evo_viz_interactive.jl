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
using WGLMakie
using Bonito
import ColorSchemes
import PDFmerger: append_pdf!

# Activate backend
WGLMakie.activate!()
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

# Define evolution condition
evo = 50
# Select fitness landscape
fit_lan = fitness_landscapes[evo]

# Define ranges of phenotypes to evaluate
x = y = z = range(-6, 6, length=100)
coords = (x, y, z)

# Evaluate fitness landscape
F = mh.fitness(coords, fit_lan)

## =============================================================================

println("Plotting example fitness landscape as a pairplot...")

# Initialize figure
fig = Figure(size=(200 * length(coords), 200 * length(coords)))

# Add grid layout
gl = fig[1, 1] = GridLayout()

# Extract indexes of axes to form lower triangular grid
axes_idx = [
    idx for idx in CartesianIndices((length(coords), length(coords)))
    if idx[1] >= idx[2]
]

# Extract dimensions
dims = DD.dims(F)

# Add axes to figure
for (i, idx) in enumerate(axes_idx)
    # Extract coordinates for this axis
    local x, y = idx.I
    # Extract dimension for this axis
    xdim = dims[x]
    ydim = dims[y]
    # Add axis to figure
    ax = Axis(gl[x, y], aspect=AxisAspect(1))
    # Hide x and y ticks
    hidedecorations!(ax)
    # Check if x == y
    if x == y
        # Compute marginal fitness landscape by summing over the other dimension
        F_marg = vec(sum(F, dims=dims[[d for d in 1:length(dims) if d != x]]))
        # Plot marginal fitness landscape
        lines!(ax, coords[x], F_marg, color=:black)
    else
        # Marginalize over the other dimension
        F_marg = sum(F, dims=dims[[d for d in 1:length(dims) if d != x && d != y]])
        # Drop dimensions that were marginalized over
        # NOTE: The dims argument must be a tuple of the dimensions to drop.
        F_marg = dropdims(
            F_marg.data,
            dims=tuple(findall(size(F_marg.data) .== 1)...)
        )
        # Plot fitness landscape
        heatmap!(ax, coords[x], coords[y], F_marg, colormap=:viridis)
        contour!(ax, coords[x], coords[y], F_marg, color=:white)
    end
end

fig

## =============================================================================

println("Plotting fitness landscapes as a 3D volume slice...")

# Initialize figure
fig = Figure(size=(600, 600))
# Add axis as a 3D scene
ax = LScene(
    fig[1, 1],
    show_axis=false,
)

# Add sliders
sgrid = SliderGrid(
    fig[2, 1],
    (label="yz plane - x axis", range=1:length(x)),
    (label="xz plane - y axis", range=1:length(y)),
    (label="xy plane - z axis", range=1:length(z)),
)

# Extract layout 
lo = sgrid.layout
# Extract number of columns
nc = ncols(lo)

# Plot volume slices
plt = volumeslices!(ax, x, y, z, F.data)

# Extract sliders
sl_yz, sl_xz, sl_xy = sgrid.sliders

# Connect sliders to `volumeslices` update methods  
on(sl_yz.value) do v
    plt[:update_yz][](v)
end
on(sl_xz.value) do v
    plt[:update_xz][](v)
end
on(sl_xy.value) do v
    plt[:update_xy][](v)
end

# Set sliders to close to the middle of the range
set_close_to!(sl_yz, 0.5 * length(x))
set_close_to!(sl_xz, 0.5 * length(y))
set_close_to!(sl_xy, 0.5 * length(z))

fig

## =============================================================================

println("Plotting fitness lanscape as contour")

# Initialize figure
fig = Figure(size=(600, 600))
# Add axis
ax = Axis3(
    fig[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    zlabel="phenotype 3",
    aspect=(1, 1, 1),
)
# Plot contour
contour!(
    ax,
    x,
    y,
    z,
    F.data,
    alpha=0.05,
    levels=7,
    colormap=:viridis,
    colorrange=[1.0, maximum(F)],
)

save("$(fig_dir)/fitness_landscape_contour.png", fig)

fig

## =============================================================================

println("Plotting fitness lanscape as contour with trajectories...")

# Initialize figure
fig = Figure(size=(600, 600))
# Add axis
ax = Axis3(
    fig[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    zlabel="phenotype 3",
    aspect=(1, 1, 1),
)

# Select trajectories evolved in example condition
fitnotype_evo = fitnotype_profiles[evo=evo, landscape=evo]

# Loop through lineages
for (i, lin) in enumerate(DD.dims(fitnotype_evo, :lineage))
    # Loop through replicate
    for (j, rep) in enumerate(DD.dims(fitnotype_evo, :replicate))
        # Extract trajectory
        traj = fitnotype_evo.phenotype[
            evo=evo, landscape=evo, lineage=lin, replicate=rep
        ]
        # Plot trajectory
        scatterlines!(
            ax,
            traj[phenotype=DD.At(:x1)].data,
            traj[phenotype=DD.At(:x2)].data,
            traj[phenotype=DD.At(:x3)].data,
            color=(ColorSchemes.glasbey_hv_n256[i], 0.5),
            markersize=4,
        )
    end # for
end # for

# Plot contour
contour!(ax, x, y, z, F, alpha=0.05, levels=minimum(F)*1.5:1:maximum(F))

save("$(fig_dir)/fitness_landscape_contour_trajectories.png", fig)

fig

## =============================================================================

println("Evaluating mutational landscape...")

# Evaluate mutational landscape
M = mh.genetic_density(coords, genetic_density)


## =============================================================================

println("Plotting mutational landscape as contour")

# Initialize figure
fig = Figure(size=(600, 600))
# Add axis
ax = Axis3(
    fig[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    zlabel="phenotype 3",
    aspect=(1, 1, 1),
)
# Plot contour
contour!(
    ax,
    x,
    y,
    z,
    M.data,
    alpha=0.05,
    levels=7,
    colormap=:magma,
)

save("$(fig_dir)/genetic_density_contour.png", fig)

fig

## =============================================================================

println("Plotting fitness and mutational landscapes in single 3D scene...")

# Initialize figure
fig = Figure(size=(600, 600))
# Add axis
ax = Axis3(
    fig[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    zlabel="phenotype 3",
    aspect=(1, 1, 1),
)

# Plot fitness landscape contour
contour!(
    ax,
    x,
    y,
    z,
    F.data,
    alpha=0.05,
    levels=5,
    colormap=:viridis,
)

# Plot mutational landscape contour
contour!(
    ax,
    x,
    y,
    z,
    M.data,
    alpha=0.05,
    levels=7,
    colormap=:magma,
)

save("$(fig_dir)/fitness_genetic_density_contour.png", fig)

fig

## =============================================================================

println("Plotting fitness and mutational landscapes with trajectories...")

# Initialize figure
fig = Figure(size=(600, 600))
# Add axis
ax = Axis3(
    fig[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    zlabel="phenotype 3",
    aspect=(1, 1, 1),
)

# Select trajectories evolved in example condition
fitnotype_evo = fitnotype_profiles[evo=evo, landscape=evo]

# Loop through lineages
for (i, lin) in enumerate(DD.dims(fitnotype_evo, :lineage))
    # Loop through replicate
    for (j, rep) in enumerate(DD.dims(fitnotype_evo, :replicate))
        # Extract trajectory
        traj = fitnotype_evo.phenotype[
            evo=evo, landscape=evo, lineage=lin, replicate=rep
        ]
        # Plot trajectory
        scatterlines!(
            ax,
            traj[phenotype=DD.At(:x1)].data,
            traj[phenotype=DD.At(:x2)].data,
            traj[phenotype=DD.At(:x3)].data,
            color=(ColorSchemes.glasbey_hv_n256[i], 0.5),
            markersize=4,
        )
    end # for
end # for

# Plot fitness landscape contour
contour!(
    ax,
    x,
    y,
    z,
    F.data,
    alpha=0.05,
    levels=7,
    colormap=:viridis,
)
# Plot mutational landscape contour
contour!(
    ax,
    x,
    y,
    z,
    M.data,
    alpha=1,
    levels=7,
    colormap=:magma,
)

save("$(fig_dir)/fitness_genetic_density_contour.png", fig)

fig