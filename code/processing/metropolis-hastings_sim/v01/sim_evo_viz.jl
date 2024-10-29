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
using CairoMakie
import PairPlots
import ColorSchemes

# Activate backend
CairoMakie.activate!()

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
mutational_landscape = JLD2.load(
    "$(sim_dir)/sim_evo.jld2"
)["mutational_landscape"]
# Load evolution condition
evolution_condition = JLD2.load(
    "$(sim_dir)/sim_evo.jld2"
)["evolution_condition"]

## =============================================================================

println("Plotting evolution fitness and mutational landscapes...")

# Define range of phenotypes to evaluate
x = range(-4, 4, length=100)
y = range(-4, 4, length=100)

# Create meshgrid
F = mh.fitness(x, y, evolution_condition)
M = mh.mutational_landscape(x, y, mutational_landscape)

# Initialize figure
fig = Figure(size=(600, 300))

# Add axis for trajectory in fitness landscape
ax1 = Axis(
    fig[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    aspect=AxisAspect(1),
    title="Fitness landscape",
)
# Add axis for trajectory in mutational landscape
ax2 = Axis(
    fig[1, 2],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    aspect=AxisAspect(1),
    title="Mutational landscape",
)

# Plot a heatmap of the fitness landscape
heatmap!(ax1, x, y, F, colormap=:viridis)
# Plot heatmap of mutational landscape
heatmap!(ax2, x, y, M, colormap=:magma)

# Plot contour plot
contour!(ax1, x, y, F, color=:white)
contour!(ax2, x, y, M, color=:white)

# Save figure
save("$(fig_dir)/evolution_fitness_mutational_landscapes.pdf", fig)
save("$(fig_dir)/evolution_fitness_mutational_landscapes.png", fig)

fig

## =============================================================================

println("Plotting evolutionary trajectories in evolution condition...")

Random.seed!(42)

# Initialize figure
fig = Figure(size=(600, 300))

# Add axis for fitness landscape
ax1 = Axis(
    fig[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    aspect=AxisAspect(1),
    title="Fitness landscape",
)
# Add axis for mutational landscape
ax2 = Axis(
    fig[1, 2],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    aspect=AxisAspect(1),
    title="Mutational landscape",
)

# Plot fitness landscape
heatmap!(ax1, x, y, F)
# Plot heatmap of mutational landscape
heatmap!(ax2, x, y, M, colormap=:magma)

# Plot contour plot
contour!(ax1, x, y, F, color=:white)
contour!(ax2, x, y, M, color=:white)

# Loop over simulations
for i in DD.dims(fitnotype_profiles, :lineage)
    # Plot trajectory
    scatterlines!.(
        [ax1, ax2],
        Ref(fitnotype_profiles.phenotype[phenotype=DD.At(:x1), lineage=i].data),
        Ref(fitnotype_profiles.phenotype[phenotype=DD.At(:x2), lineage=i].data),
        color=ColorSchemes.seaborn_colorblind[i],
        markersize=3
    )
end

# Set limits
xlims!(ax1, -4, 4)
ylims!(ax1, -4, 4)
xlims!(ax2, -4, 4)
ylims!(ax2, -4, 4)

# Save figure
save("$(fig_dir)/evolution_condition_trajectories.pdf", fig)
save("$(fig_dir)/evolution_condition_trajectories.png", fig)

fig

## =============================================================================

println("Plotting example alternative fitness landscapes...")

# Define number of rows and columns
n_rows = 5
n_cols = 5

# Define ranges of phenotypes to evaluate
x = range(-6, 6, length=100)
y = range(-6, 6, length=100)

# Initialize figure
fig = Figure(size=(200 * n_cols, 200 * n_rows))

# Add grid layout
gl = fig[1, 1] = GridLayout()

# Loop over fitness landscapes
for i in 1:(n_rows*n_cols)
    # Extract fitness landscape
    fit_lan = fitness_landscapes[i]
    # Define row and column
    row = (i - 1) ÷ n_cols + 1
    col = (i - 1) % n_cols + 1
    # Add axis
    ax = Axis(gl[row, col], aspect=AxisAspect(1))
    # Evaluate fitness landscape
    F = mh.fitness(x, y, fit_lan)
    # Plot fitness landscape
    heatmap!(ax, x, y, F, colormap=:viridis)
    # Plot contour plot
    contour!(ax, x, y, F, color=:white)
end

# Add global x and y labels
Label(gl[end+1, :], "phenotype 1")
Label(gl[:, 0], "phenotype 2", rotation=π / 2)

# Save figure
save("$(fig_dir)/alternative_fitness_landscapes.pdf", fig)
save("$(fig_dir)/alternative_fitness_landscapes.png", fig)

fig

## =============================================================================

println("Plotting fitness profiles for sample time points...")

# Define number of time points to plot
n_tps_plot = 4

# Define time point indices to plot as evenly spaced as possible
tps_plot = Int.(range(
    DD.dims(fitnotype_profiles, :time)[[1, end]]..., length=n_tps_plot
))

# Initialize figure
fig = Figure(size=(400, 150 * n_tps_plot))

# Add grid layout for entire figure
gl = fig[1, 1] = GridLayout()

# Add grid layout for plots
gl_plots = gl[1:5, 1:5] = GridLayout()

# Loop over time points
for (i, tp) in enumerate(tps_plot)
    # Add axis
    ax = Axis(
        gl_plots[i, 1],
        title="t = $tp",
        yscale=log2,
    )
    # Check if final plot
    if i ≠ n_tps_plot
        # Turn off x-axis
        hidexdecorations!(ax, grid=false)
    end
    # Loop over lineages
    for lin in DD.dims(fitnotype_profiles, :lineage)
        # Plot fitness profile
        scatterlines!(
            ax,
            collect(DD.dims(fitnotype_profiles, :landscape)),
            fitnotype_profiles.fitness[time=DD.At(tp), lineage=lin].data,
            color=ColorSchemes.glasbey_hv_n256[lin],
            markersize=6
        )
    end # for 
end # for i

# Add global x and y labels
Label(gl[end+1, 3], "environment index")
Label(gl[3, 0], "fitness", rotation=π / 2)

# Save figure
save("$(fig_dir)/fitness_profiles.pdf", fig)
save("$(fig_dir)/fitness_profiles.png", fig)

fig

## =============================================================================

println("Performing SVD on standardized fitness profiles...")

# Reshape the array to stack the 3rd dimension
fit_mat = log.(
    reshape(fitnotype_profiles.fitness.data, size(fitnotype_profiles, 4), :)
)

# Fit model to standardize data to mean zero and standard deviation 1 on each
# environment 
dt = StatsBase.fit(StatsBase.ZScoreTransform, fit_mat, dims=2)

# Standardize the data to have mean 0 and standard deviation 1
fit_std = StatsBase.transform(dt, fit_mat)

# Compute SVD
U, S, V = LinearAlgebra.svd(fit_std);

## =============================================================================

println("Plotting Singular Value Spectrum...")

# Initialize figure
fig = Figure(size=(650, 300))

# Add axis for singular values
ax1 = Axis(
    fig[1, 1],
    title="Singular Values",
    xlabel="singular value index",
    ylabel="singular value",
)

# Plot singular values
scatterlines!(ax1, S)

# Add axis for percentage of variance explained
ax2 = Axis(
    fig[1, 2],
    title="Fraction of Variance Explained",
    xlabel="principal component index",
    ylabel="fraction of variance explained",
)
# Compute percentage of variance explained
pve = S .^ 2 ./ sum(S .^ 2)
# Plot percentage of variance explained
scatterlines!(ax2, pve)

# Save figure
save("$(fig_dir)/singular_value_spectrum.pdf", fig)
save("$(fig_dir)/singular_value_spectrum.png", fig)

fig

## =============================================================================

println("Plotting 2D projection of standardized fitness profiles...")

# Project data onto first two principal components
fit_pca = U[:, 1:2]' * fit_std

# Initialize figure
fig = Figure(size=(300, 300))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="principal component 1",
    ylabel="principal component 2",
    aspect=AxisAspect(1),
)

# Plot fitness profiles
scatter!(ax, fit_pca[1, :], fit_pca[2, :], markersize=5)

# Save figure
save("$(fig_dir)/fitness_profiles_2DPCA_projection.pdf", fig)
save("$(fig_dir)/fitness_profiles_2DPCA_projection.png", fig)

fig

## =============================================================================

println("Plotting comparison between phenotype and PCA-projected fitness trajectories...")

# Standardize each slice of the fitnotype profiles
fit_pca_std = StatsBase.transform.(
    Ref(dt), eachslice(log.(fitnotype_profiles.fitness.data), dims=3)
)

# Initialize figure
fig = Figure(size=(600, 300))

# Add axis for original space
ax1 = Axis(
    fig[1, 1],
    title="Phenotype space",
    aspect=AxisAspect(1),
    xlabel="phenotype 1",
    ylabel="phenotype 2",
)

# Add axis for PCA space
ax2 = Axis(
    fig[1, 2],
    title="PCA space",
    aspect=AxisAspect(1),
    xlabel="principal component 1",
    ylabel="principal component 2",
)


# Loop over lineages
for lin in DD.dims(fitnotype_profiles, :lineage)
    # Plot trajectory
    scatterlines!(
        ax1,
        fitnotype_profiles.phenotype[phenotype=DD.At(:x1), lineage=lin].data,
        fitnotype_profiles.phenotype[phenotype=DD.At(:x2), lineage=lin].data,
        color=ColorSchemes.seaborn_colorblind[lin],
        markersize=4
    )
end

# Loop through each simulation (2nd dimension)
for (j, slice) in enumerate(fit_pca_std)
    # Project slice onto PCA space
    pca_slice = U[:, 1:2]' * slice
    # Plot slice
    scatterlines!(
        ax2,
        pca_slice[1, :],
        pca_slice[2, :],
        color=ColorSchemes.seaborn_colorblind[j],
        markersize=4
    )
end

# Save figure
save("$(fig_dir)/fitness_trajectories_phenotype_PCA_projection.pdf", fig)
save("$(fig_dir)/fitness_trajectories_phenotype_PCA_projection.png", fig)

fig

## =============================================================================