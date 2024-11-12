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
mutational_landscape = JLD2.load(
    "$(sim_dir)/sim_evo.jld2"
)["mutational_landscape"]

## =============================================================================

println("Plotting fitness landscapes...")

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
pdf_path = "$(fig_dir)/fitness_landscapes.pdf"

for page in 1:n_pages
    # Initialize figure for this page
    local fig = Figure(size=(200 * n_cols, 200 * n_rows))
    gl = fig[1, 1] = GridLayout()

    # Calculate range of landscapes for this page
    start_idx = (page - 1) * landscapes_per_page + 1
    end_idx = min(page * landscapes_per_page, length(fitness_landscapes))

    # Loop over fitness landscapes for this page
    for (i, idx) in enumerate(start_idx:end_idx)
        fit_lan = fitness_landscapes[idx]
        row = (i - 1) ÷ n_cols + 1
        col = (i - 1) % n_cols + 1

        local ax = Axis(gl[row, col], aspect=AxisAspect(1))
        F = mh.fitness(x, y, fit_lan)
        heatmap!(ax, x, y, F, colormap=:viridis)
        contour!(ax, x, y, F, color=:white)
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

# Initialize figure
fig = Figure(size=(300, 300))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    aspect=AxisAspect(1),
)

# Evaluate mutational landscape
M = mh.mutational_landscape(x, y, mutational_landscape)

# Plot mutational landscape
heatmap!(ax, x, y, M, colormap=:magma)

# Add contour plot
contour!(ax, x, y, M, color=:white)

# Save figure
save("$(fig_dir)/mutational_landscape.pdf", fig)
save("$(fig_dir)/mutational_landscape.png", fig)

fig

## =============================================================================

println("Plotting mutational landscape with initial conditions...")

# Initialize figure
fig = Figure(size=(300, 300))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    aspect=AxisAspect(1),
)

# Evaluate mutational landscape
M = mh.mutational_landscape(x, y, mutational_landscape)

# Plot mutational landscape
heatmap!(ax, x, y, M, colormap=:magma)

# Add contour plot
contour!(ax, x, y, M, color=:white)

# Plot initial conditions
scatter!(
    ax,
    vec(fitnotype_profiles.phenotype[time=1, evo=1, phenotype=DD.At(:x1)]),
    vec(fitnotype_profiles.phenotype[time=1, evo=1, phenotype=DD.At(:x2)]),
    markersize=8,
    marker=:star5,
)

# Save figure
save("$(fig_dir)/mutational_landscape_initial_conditions.pdf", fig)
save("$(fig_dir)/mutational_landscape_initial_conditions.png", fig)

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

# Define number of replicates
n_rep = length(DD.dims(fitnotype_profiles, :replicate))

for page in 1:n_pages
    # Initialize figure for this page
    local fig = Figure(size=(200 * n_cols, 200 * n_rows))
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
        local ax = Axis(gl[row, col], aspect=AxisAspect(1))

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
                    fitnotype_profiles.phenotype[
                        phenotype=DD.At(:x1),
                        lineage=lin,
                        replicate=rep,
                        landscape=idx,
                        evo=idx,
                    ].data,
                    fitnotype_profiles.phenotype[
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
    local fig = Figure(size=(200 * n_cols, 200 * n_rows))
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
        local ax = Axis(gl[row, col], aspect=AxisAspect(1))

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
                    fitnotype_profiles.phenotype[
                        time=DD.At(1:n_sub:300),
                        phenotype=DD.At(:x1),
                        lineage=lin,
                        replicate=rep,
                        landscape=idx,
                        evo=idx,
                    ].data,
                    fitnotype_profiles.phenotype[
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

println("Plotting evolutionary trajectories in evolution condition showing outline of mutational landscape...")

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
pdf_path = "$(fig_dir)/evolution_condition_trajectories_mutational_landscape.pdf"

for page in 1:n_pages
    # Initialize figure for this page
    local fig = Figure(size=(200 * n_cols, 200 * n_rows))
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
        local ax = Axis(gl[row, col], aspect=AxisAspect(1))

        # Evaluate fitness landscape
        F = mh.fitness(x, y, fit_lan)
        # Plot fitness landscape
        heatmap!(ax, x, y, F, colormap=:viridis)
        # Plot contour plot
        contour!(ax, x, y, F, color=:white)
        # Plot mutational landscape contours
        contour!(ax, x, y, M, color=:black, linestyle=(:dash, :dense))

        # Set limits
        xlims!(ax, -4, 4)
        ylims!(ax, -4, 4)

        # Loop over simulations
        for lin in DD.dims(fitnotype_profiles, :lineage)
            # Loop over replicates
            for rep in DD.dims(fitnotype_profiles, :replicate)
                # Extract x and y coordinates
                x_data = fitnotype_profiles.phenotype[
                    time=DD.At(1:300),
                    phenotype=DD.At(:x1),
                    lineage=lin,
                    replicate=rep,
                    landscape=idx,
                    evo=idx,
                ].data
                y_data = fitnotype_profiles.phenotype[
                    time=DD.At(1:300),
                    phenotype=DD.At(:x2),
                    lineage=lin,
                    replicate=rep,
                    landscape=idx,
                    evo=idx,
                ].data

                # Plot trajectory
                lines!(
                    ax,
                    x_data,
                    y_data,
                    color=ColorSchemes.glasbey_hv_n256[(lin-1)*n_rep+rep],
                    linewidth=1
                )

                # Plot initial condition
                scatter!(
                    ax,
                    x_data[1],
                    y_data[1],
                    color=ColorSchemes.glasbey_hv_n256[(lin-1)*n_rep+rep],
                    markersize=8,
                    marker=:xcross,
                )

                # Plot final condition
                scatter!(
                    ax,
                    x_data[end],
                    y_data[end],
                    color=ColorSchemes.glasbey_hv_n256[(lin-1)*n_rep+rep],
                    markersize=8,
                    marker=:utriangle,
                )
            end # for rep
        end # for lin
    end # for page

    # Add global x and y labels
    Label(gl[end+1, :], "phenotype 1")
    Label(gl[:, 0], "phenotype 2", rotation=π / 2)

    # Save as PNG
    save(
        "$(fig_dir)/evolution_condition_trajectories_mutational_landscape_$(lpad(page, 2, '0')).png",
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
    title="Fraction of Variance Explained",
    xlabel="principal component index",
    ylabel="fraction of variance explained",
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