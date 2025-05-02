## =============================================================================

println("Importing packages...")

# Import project package
import Antibiotic
import Antibiotic.mh as mh

# Import packages for storing results
import DimensionalData as DD

# Import JLD2 for saving results
import JLD2

# Import basic math libraries
import StatsBase
import MultivariateStats as MStats
import LinearAlgebra
import Random
import Distributions
import Distances
# Load CairoMakie for plotting
using CairoMakie
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

# Extract initial and final time points
t_init, t_final = collect(DD.dims(fitnotype_profiles, :time)[[1, end]])
# Define step size
t_step = DD.dims(fitnotype_profiles, :time)[2] - t_init
# Define by how much to subsample the time series
n_sub = 10

# Subsample time series
fitnotype_profiles = fitnotype_profiles[time=DD.At(t_init:n_sub*t_step:t_final)]

# Define number of environments
n_env = length(DD.dims(fitnotype_profiles, :landscape))

# Extract fitness data bringing the fitness dimension to the first dimension
fit_data = permutedims(fitnotype_profiles.fitness.data, (5, 1, 2, 3, 4, 6))
# Reshape the array to a Matrix
fit_data = reshape(fit_data, size(fit_data, 1), :)

# Reshape the array to stack the 3rd dimension
fit_mat = log.(fit_data)

# Fit model to standardize data to mean zero and standard deviation 1 on each
# environment 
dt = StatsBase.fit(StatsBase.ZScoreTransform, fit_mat, dims=2)

# Standardize the data to have mean 0 and standard deviation 1
fit_std = StatsBase.transform(dt, fit_mat)

# Standardize the data to have mean 0 and standard deviation 1
log_fitnotype_std = DD.DimArray(
    mapslices(slice -> StatsBase.transform(dt, slice),
        log.(fitnotype_profiles.fitness.data),
        dims=[5]),
    fitnotype_profiles.fitness.dims,
)

## =============================================================================

println("Defining ranges of phenotypes to evaluate...")

# Define ranges of phenotypes to evaluate
x = range(-6, 6, length=100)
y = range(-6, 6, length=100)

## =============================================================================
println("Plotting fitness landscapes...")

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

        local ax = Axis(
            gl[row, col],
            aspect=AxisAspect(1),
            yticklabelsvisible=false,
            xticklabelsvisible=false,
        )
        F = mh.fitness(x, y, fit_lan)
        heatmap!(ax, x, y, F, colormap=:algae)
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
    xticklabelsvisible=false,
    yticklabelsvisible=false,
)

# Evaluate mutational landscape
M = mh.genetic_density(x, y, genetic_density)

# Plot mutational landscape
heatmap!(ax, x, y, M, colormap=Reverse(ColorSchemes.Purples_9))

# Add contour plot
contour!(ax, x, y, M, color=:white)

# Save figure
save("$(fig_dir)/genetic_density.pdf", fig)
save("$(fig_dir)/genetic_density.png", fig)

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
    xticklabelsvisible=false,
    yticklabelsvisible=false,
)

# Evaluate mutational landscape
M = mh.genetic_density(x, y, genetic_density)

# Plot mutational landscape
heatmap!(ax, x, y, M, colormap=Reverse(ColorSchemes.Purples_9))

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
save("$(fig_dir)/genetic_density_initial_conditions.pdf", fig)
save("$(fig_dir)/genetic_density_initial_conditions.png", fig)

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
        local ax = Axis(
            gl[row, col],
            aspect=AxisAspect(1),
            xticklabelsvisible=false,
            yticklabelsvisible=false,
        )

        # Evaluate fitness landscape
        F = mh.fitness(x, y, fit_lan)
        # Plot fitness landscape
        heatmap!(ax, x, y, F, colormap=:algae)
        # Plot contour plot
        contour!(
            ax,
            x,
            y,
            F,
            color=Antibiotic.viz.colors()[:black],
            linestyle=:dash,
        )
        # Plot mutational landscape contours
        contour!(ax, x, y, M, color=:white)

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

# Plot all of the data as a cloud of points colored by lineage

# Initialize figure
fig = Figure(size=(300, 300))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    aspect=AxisAspect(1)
)

# Define sampling interval
n_sub = 10

# Loop over simulations
for (i, lin) in enumerate(DD.dims(fitnotype_profiles, :lineage))
    # Loop over replicates
    for rep in DD.dims(fitnotype_profiles, :replicate)
        for evo in DD.dims(fitnotype_profiles, :evo)
            # Plot latent space
            scatter!(
                ax,
                vec(fitnotype_profiles.phenotype[
                    time=DD.At(t_init:n_sub*t_step:t_final),
                    phenotype=DD.At(:x1),
                    lineage=lin,
                    replicate=rep,
                    landscape=DD.At(1),
                    evo=evo,
                ]),
                vec(fitnotype_profiles.phenotype[
                    time=DD.At(t_init:n_sub*t_step:t_final),
                    phenotype=DD.At(:x2),
                    lineage=lin,
                    replicate=rep,
                    landscape=DD.At(1),
                    evo=evo,
                ]),
                color=(ColorSchemes.glasbey_hv_n256[i], 0.25),
                rasterize=true,
            )
        end # for evo
    end # for rep
end # for lin

save("$(fig_dir)/scatter_by_lineage.png", fig)

fig


## =============================================================================

println("Performing SVD on standardized fitness profiles...")

# Compute SVD
U, S, V = LinearAlgebra.svd(fit_std)

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

# Perform PCA on the data 
fit_pca = MStats.fit(MStats.PCA, fit_std, maxoutdim=2)

# Project data onto first two principal components
fit_pca_coords = MStats.predict(fit_pca, fit_std)

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
scatter!(ax, Point2f.(eachcol(fit_pca_coords)), markersize=5)

# Save figure
save("$(fig_dir)/fitness_profiles_2DPCA_projection.png", fig)

fig

## =============================================================================