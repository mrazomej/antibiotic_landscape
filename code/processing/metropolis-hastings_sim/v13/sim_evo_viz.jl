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
import PDFmerger: append_pdf!

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
        heatmap!(ax, coords[x], coords[y], F_marg, colormap=:algae)
        contour!(ax, coords[x], coords[y], F_marg, color=:white)
    end
end

# Save figure
save("$(fig_dir)/fitness_landscape_pairplot.pdf", fig)
save("$(fig_dir)/fitness_landscape_pairplot.png", fig)

fig

## =============================================================================

println("Plotting fitness landscapes with 2D panels...")

# Define ranges of phenotypes to evaluate
x = range(-6, 6, length=50)
y = range(-6, 6, length=50)
z = range(-6, 6, length=50)
coords = (x, y, z)

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

# Define number of rows and columns per page
n_rows = 5
n_cols = 5
landscapes_per_page = n_rows * n_cols

# Calculate number of pages needed
n_pages = ceil(Int, length(fitness_landscapes) / landscapes_per_page)

# Initialize PDF for appending
pdf_path = "$(fig_dir)/fitness_landscapes.pdf"

# Remove pdf file if it exists
if isfile(pdf_path)
    rm(pdf_path)
end

for page in 1:n_pages
    # Initialize figure for this page
    fig = Figure(size=(200 * n_cols, 200 * n_rows))
    gl = fig[1, 1] = GridLayout()

    # Calculate range of landscapes for this page
    start_idx = (page - 1) * landscapes_per_page + 1
    end_idx = min(page * landscapes_per_page, length(fitness_landscapes))

    # Loop over fitness landscapes for this page
    for (i, idx) in enumerate(start_idx:end_idx)
        # Select fitness landscape
        fit_lan = fitness_landscapes[idx]
        # Define row and column
        row = (i - 1) ÷ n_cols + 1
        col = (i - 1) % n_cols + 1
        # Add axis
        ax = Axis3(
            gl[row, col],
            aspect=(1, 1, 1),
            xgridvisible=false,
            ygridvisible=false,
            zgridvisible=false,
        )
        # Hide decorations
        hidedecorations!(ax)
        # Evaluate fitness landscape
        F = mh.fitness(coords, fit_lan)

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
    end

    # Add global x and y labels
    Label(gl[end+1, :], "phenotype 1")
    Label(gl[:, 0], "phenotype 2", rotation=π / 2)

    # Save as PNG
    save("$(fig_dir)/fitness_landscapes_$(lpad(page, 2, '0')).png", fig)

    # Save or append to PDF
    save("temp.pdf", fig)
    append_pdf!(pdf_path, "temp.pdf", cleanup=true)
end

## =============================================================================

println("Plotting mutational landscape...")

# Define ranges of phenotypes to evaluate
x = y = z = range(-6, 6, length=50)
coords = (x, y, z)


# Initialize figure
fig = Figure(size=(300, 300))

# Add axis
ax = Axis3(
    fig[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    zlabel="genetic density",
    aspect=(1, 1, 1),
)

# Evaluate mutational landscape
M = mh.genetic_density(coords, genetic_density)

# Plot mutational landscape
contour!(ax, x, y, z, M)

# Save figure
save("$(fig_dir)/genetic_density.pdf", fig)
save("$(fig_dir)/genetic_density.png", fig)

fig

## =============================================================================

println("Plotting evolutionary trajectories in evolution condition...")

# Define ranges of phenotypes to evaluate
x = range(-6, 6, length=100)
y = range(-6, 6, length=100)

# Define number of rows and columns per page
n_rows = 5
n_cols = 5
landscapes_per_page = n_rows * n_cols

# Calculate number of pages needed
n_pages = ceil(Int, length(fitness_landscapes) / landscapes_per_page)

# Initialize PDF for appending
pdf_path = "$(fig_dir)/evolution_condition_trajectories.pdf"

for page in 1:n_pages
    # Initialize figure for this page
    fig = Figure(size=(200 * n_cols, 200 * n_rows))
    gl = fig[1, 1] = GridLayout()

    # Calculate range of landscapes for this page
    start_idx = (page - 1) * landscapes_per_page + 1
    end_idx = min(page * landscapes_per_page, length(fitness_landscapes))

    # Loop over fitness landscapes for this page
    for (i, idx) in enumerate(start_idx:end_idx)
        # Extract fitness landscape
        fit_lan = fitness_landscapes[idx]
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

        # Set limits
        xlims!(ax, -4, 4)
        ylims!(ax, -4, 4)

        # Loop over simulations
        for lin in DD.dims(fitnotype_profiles, :lineage)
            # Loop over replicates
            for rep in DD.dims(fitnotype_profiles, :replicate)
                # Plot trajectory
                lines!(
                    ax,
                    x_traj.phenotype[
                        phenotype=DD.At(:x1),
                        lineage=lin,
                        replicate=rep,
                        landscape=idx,
                        evo=idx,
                    ].data,
                    x_traj.phenotype[
                        phenotype=DD.At(:x2),
                        lineage=lin,
                        replicate=rep,
                        landscape=idx,
                        evo=idx,
                    ].data,
                    color=ColorSchemes.glasbey_hv_n256[(lin-1)*n_rep+rep],
                    linewidth=1
                )
            end # for rep
        end # for lin
    end # for page

    # Add global x and y labels
    Label(gl[end+1, :], "phenotype 1")
    Label(gl[:, 0], "phenotype 2", rotation=π / 2)

    # Save as PNG
    save("$(fig_dir)/evolution_condition_trajectories_$(lpad(page, 2, '0')).png", fig)

    # Save or append to PDF
    save("temp.pdf", fig)
    append_pdf!(pdf_path, "temp.pdf", cleanup=true)
end

## =============================================================================

println("Plotting evolutionary trajectories in evolution condition subsampling the time series...")

# Define number of time points to subsample
n_sub = 10

# Define ranges of phenotypes to evaluate
x = range(-6, 6, length=100)
y = range(-6, 6, length=100)

# Define number of rows and columns per page
n_rows = 5
n_cols = 5
landscapes_per_page = n_rows * n_cols

# Calculate number of pages needed
n_pages = ceil(Int, length(fitness_landscapes) / landscapes_per_page)

# Initialize PDF for appending
pdf_path = "$(fig_dir)/evolution_condition_trajectories_subsampled.pdf"

for page in 1:n_pages
    # Initialize figure for this page
    fig = Figure(size=(200 * n_cols, 200 * n_rows))
    gl = fig[1, 1] = GridLayout()

    # Calculate range of landscapes for this page
    start_idx = (page - 1) * landscapes_per_page + 1
    end_idx = min(page * landscapes_per_page, length(fitness_landscapes))

    # Loop over fitness landscapes for this page
    for (i, idx) in enumerate(start_idx:end_idx)
        # Extract fitness landscape
        fit_lan = fitness_landscapes[idx]
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

        # Set limits
        xlims!(ax, -4, 4)
        ylims!(ax, -4, 4)

        # Loop over simulations
        for lin in DD.dims(fitnotype_profiles, :lineage)
            # Loop over replicates
            for rep in DD.dims(fitnotype_profiles, :replicate)
                # Plot trajectory
                lines!(
                    ax,
                    x_traj.phenotype[
                        time=DD.At(1:n_sub:300),
                        phenotype=DD.At(:x1),
                        lineage=lin,
                        replicate=rep,
                        landscape=idx,
                        evo=idx,
                    ].data,
                    x_traj.phenotype[
                        time=DD.At(1:n_sub:300),
                        phenotype=DD.At(:x2),
                        lineage=lin,
                        replicate=rep,
                        landscape=idx,
                        evo=idx,
                    ].data,
                    color=ColorSchemes.glasbey_hv_n256[(lin-1)*n_rep+rep],
                    linewidth=1
                )
            end # for rep
        end # for lin
    end # for page

    # Add global x and y labels
    Label(gl[end+1, :], "phenotype 1")
    Label(gl[:, 0], "phenotype 2", rotation=π / 2)

    # Save as PNG
    save(
        "$(fig_dir)/evolution_condition_trajectories_subsampled_$(lpad(page, 2, '0')).png",
        fig
    )

    # Save or append to PDF
    save("temp.pdf", fig)
    append_pdf!(pdf_path, "temp.pdf", cleanup=true)
end

## =============================================================================

println("Performing SVD on standardized fitness profiles...")

# Reshape the array to stack the 3rd dimension
fit_mat = log.(
    reshape(
        fitnotype_profiles.fitness[time=DD.At(1:10:300)].data,
        size(fitnotype_profiles, 5),
        :
    )
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
    title="% Variance Explained",
    xlabel="principal component index",
    ylabel="% variance explained",
)
# Compute percentage of variance explained
pve = S .^ 2 ./ sum(S .^ 2)
# Plot percentage of variance explained
scatterlines!(ax2, pve)

# Save figure
save("$(fig_dir)/singular_value_spectrum.png", fig)
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