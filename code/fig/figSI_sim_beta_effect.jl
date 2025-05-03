## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic
import Antibiotic.mh as mh

# Import AutoEncoderToolkit to train VAEs
import AutoEncoderToolkit as AET

# Import libraries to handle data
import Glob
import DimensionalData as DD
import DataFrames as DF

# Import library to save models
import JLD2

# Import IterTools for Cartesian product
import IterTools

# Import basic math
import StatsBase
import Random
import LinearAlgebra
import Distributions

# Load Plotting packages
using CairoMakie
using Makie
import ColorSchemes
import Colors
# Activate backend
CairoMakie.activate!()

# Set plotting style
Antibiotic.viz.theme_makie!()

# Set random seed
Random.seed!(42)

## =============================================================================


println("Defining directories...")

# Define output directory
fig_dir = "$(git_root())/fig/supplementary"

# Generate output directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating output directory...")
    mkpath(fig_dir)
end

## =============================================================================

println("Defining evolution parameters...")

# Phenotype space dimensionality
n_dim = 2
# Number of initial conditions (positions in phenotype space)
n_sim = 10
# Number of replicates (evolving strains per initial condition)
n_rep = 2
# Inverse temperature
β = [1.0, 10.0, 100.0]
# mutation step size
µ = 0.1
# Number of evolution steps
n_steps = 300

# Define range of peak means
peak_mean_min = -4.0
peak_mean_max = 4.0

## =============================================================================

println("Defining SPECIFIC genotype-phenotype density landscape...")

# Mutational peak amplitude
mut_evo_amplitude = 1.0
# Mutational peak means
mut_means = [
    [-1.5, -1.5],
    [1.5, -1.5],
    [1.5, 1.5],
    [-1.5, 1.5],
]
# Mutational peak covariance
mut_evo_covariance = 0.45
# Create mutational peaks
mut_evo_peaks = mh.GaussianPeaks(
    mut_evo_amplitude,
    mut_means,
    mut_evo_covariance
)

# Define grid on which to evaluate mutational landscape
mut_evo_grid = range(peak_mean_min, peak_mean_max, length=100)

# Evaluate mutational landscape on grid
mut_evo_grid_points = mh.genetic_density(
    tuple(repeat([mut_evo_grid], n_dim)...),
    mut_evo_peaks
)

# Define grid of possible initial conditions
init_grid = [[x...] for x in IterTools.product(fill(mut_evo_grid, 2)...)]

## =============================================================================

println("Defining FIXED evolution condition...")

# Evolution condition amplitude
fit_evo_amplitude = 5.0
# Evolution condition mean
fit_evo_mean = [0.0, 0.0]
# Evolution condition covariance
fit_evo_covariance = 3.0
# Create peak
fit_evo_peak = mh.GaussianPeak(
    fit_evo_amplitude,
    fit_evo_mean,
    fit_evo_covariance
)

## =============================================================================

Random.seed!(42)

println("Simulating evolution and computing fitnotype profiles...")

# Sample initial positions on phenotype space from uniform distribution taking
# into account mutational landscape (not to start at a low mutational peak)
x0 = StatsBase.sample(
    vec(init_grid),
    StatsBase.Weights(vec(mut_evo_grid_points)),
    n_sim
)

# Select initial conditions for replicates from multivariate normal distribution
# around each initial condition
x0_reps = reduce(
    (x, y) -> cat(x, y, dims=3),
    [
        rand(Distributions.MvNormal(x0[i], 0.1), n_rep)
        for i in 1:n_sim
    ]
)
# Change order of dimensions
x0_reps = permutedims(x0_reps, (1, 3, 2))

# Define dimensions to be used with DimensionalData
phenotype = DD.Dim{:phenotype}([:x1, :x2]) # phenotype
fitness = DD.Dim{:fitness}([:fitness]) # fitness
time = DD.Dim{:time}(0:n_steps) # time
lineage = DD.Dim{:lineage}(1:n_sim) # lineage
replicate = DD.Dim{:replicate}(1:n_rep) # replicate
inv_temp = DD.Dim{:inv_temp}(β) # inverse temperature


# Initialize DimensionalData array to hold trajectories and fitness
phenotype_traj = DD.zeros(
    Float32,
    phenotype,
    time,
    lineage,
    replicate,
    inv_temp,
)

fitness_traj = DD.zeros(
    Float32,
    fitness,
    time,
    lineage,
    replicate,
    inv_temp,
)


# Stack arrays to store trajectories in phenotype and fitness dimensions
x_traj = DD.DimStack(
    (phenotype=phenotype_traj, fitness=fitness_traj),
)

# Store initial conditions
x_traj.phenotype[time=DD.At(1)] = repeat(
    x0_reps,
    outer=(1, 1, 1, length(β))
)

# Map initial phenotype to fitness
x_traj.fitness[time=DD.At(1)] = repeat(
    mh.fitness(
        x_traj.phenotype[time=DD.At(1), inv_temp=DD.At(1.0)].data,
        fit_evo_peak
    ),
    outer=(1, 1, 1, length(β))
)

# Loop over inverse temperatures
for (i, inv_temp) in enumerate(DD.dims(x_traj, :inv_temp))
    # Loop over lineages
    for lin in DD.dims(x_traj, :lineage)
        # Loop over replicates
        for rep in DD.dims(x_traj, :replicate)
            # Run Metropolis-Hastings algorithm
            trajectory = mh.evo_metropolis_hastings(
                x_traj.phenotype[
                    time=DD.At(1),
                    lineage=DD.At(lin),
                    replicate=DD.At(rep),
                    inv_temp=DD.At(inv_temp),
                ].data,
                fit_evo_peak,
                mut_evo_peaks,
                β[i],
                µ,
                n_steps
            )

            # Store trajectory
            x_traj.phenotype[
                lineage=DD.At(lin),
                replicate=DD.At(rep),
                inv_temp=DD.At(inv_temp),
            ] = trajectory

            # Calculate and store fitness for each point in the trajectory
            x_traj.fitness[
                lineage=DD.At(lin),
                replicate=DD.At(rep),
                inv_temp=DD.At(inv_temp),
            ] = mh.fitness(trajectory, fit_evo_peak)
        end # for
    end # for
end # for

## =============================================================================

# Initialize figure
fig = Figure(size=(250 * length(β), 500))

# Add grid layout
gl = GridLayout(fig[1, 1])

# Add grid layout for banner
gl_banner = gl[1, 1] = GridLayout()

# Add grid layout for fitness landscape
gl_fitness = gl[2, 1] = GridLayout()

# Add grid layout for time series
gl_time = gl[3, 1] = GridLayout()

# Adjust row size
rowsize!(gl, 2, Auto(1))
rowsize!(gl, 3, Auto(3 / 4))

# ------------------------------------------------------------------------------

# Add box for section title
Box(
    gl_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-50) # Moves box to the left and right
)

# Add section title
Label(
    gl_banner[1, 1],
    "inverse temperature effect on adaptive dynamics",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
    alignmode=Mixed(; left=-30) # Moves text to the left
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

# Create meshgrid
F = mh.fitness(x, y, fit_evo_peak)
G = mh.genetic_density(x, y, mut_evo_peaks)

# Loop over inverse temperatures
for (i, inv_temp) in enumerate(DD.dims(x_traj, :inv_temp))
    # Add axis for fitness landscape
    ax_landscape = Axis(
        gl_fitness[1, i],
        xlabel="phenotype 1",
        ylabel="phenotype 2",
        aspect=AxisAspect(1),
        yticklabelsvisible=false,
        xticklabelsvisible=false,
        title="β = $(β[i])",
    )

    # Plot fitness landscape
    heatmap!(ax_landscape, x, y, F, colormap=:algae)

    # Plot contour lines for fitness landscape
    contour!(ax_landscape, x, y, F, color=:black, linestyle=:dash)

    # Plot contour lines for genetic density
    contour!(ax_landscape, x, y, G, color=:white, linestyle=:solid)

    # Add axis for time series
    ax_time = Axis(
        gl_time[1, i],
        xlabel="time",
        ylabel="fitness",
        aspect=AxisAspect(4 / 3),
        yticklabelsvisible=false,
        xticklabelsvisible=false,
    )

    # Initialize counter
    counter = 1
    # Loop over lineages
    for lin in DD.dims(x_traj, :lineage)
        # Loop over replicates
        for rep in DD.dims(x_traj, :replicate)
            # Plot trajectory
            lines!(
                ax_landscape,
                x_traj.phenotype[
                    phenotype=DD.At(:x1),
                    lineage=lin,
                    replicate=rep,
                    inv_temp=DD.At(inv_temp),
                ].data,
                x_traj.phenotype[
                    phenotype=DD.At(:x2),
                    lineage=lin,
                    replicate=rep,
                    inv_temp=DD.At(inv_temp),
                ].data,
                color=ColorSchemes.glasbey_hv_n256[counter],
                linewidth=2
            )
            # Plot initial and final points
            point_init = scatter!(
                ax_landscape,
                Point2f(x_traj.phenotype[
                    lineage=DD.At(lin),
                    replicate=DD.At(rep),
                    time=DD.At(1),
                    inv_temp=DD.At(inv_temp),
                ]),
                color=ColorSchemes.glasbey_hv_n256[counter],
                markersize=8,
                marker=:xcross,
                strokecolor=:black,
                strokewidth=1,
            )
            point_final = scatter!(
                ax_landscape,
                Point2f(x_traj.phenotype[
                    lineage=DD.At(lin),
                    replicate=DD.At(rep),
                    time=DD.At(last(DD.dims(x_traj, :time))),
                    inv_temp=DD.At(inv_temp),
                ]),
                color=ColorSchemes.glasbey_hv_n256[counter],
                markersize=10,
                marker=:utriangle,
                strokecolor=:black,
                strokewidth=1.5,
            )
            # Translate to front of other trajectories
            translate!(point_init, 0, 0, 100)
            translate!(point_final, 0, 0, 100)

            # Plot fitness trajectory
            lines!(
                ax_time,
                vec(x_traj.fitness[
                    lineage=lin,
                    replicate=rep,
                    inv_temp=DD.At(inv_temp),
                ].data),
                color=ColorSchemes.glasbey_hv_n256[counter],
            )
            # Increment counter
            counter += 1
        end # for rep
    end # for lin

    # Set axis limits
    xlims!(ax_landscape, phenotype_lims.x...)
    ylims!(ax_landscape, phenotype_lims.y...)

    ylims!(
        ax_time,
        minimum(x_traj.fitness.data) - 0.25,
        maximum(x_traj.fitness.data) + 0.25
    )
end # for

# Save figure
save("$(fig_dir)/figSI_sim_beta_effect.pdf", fig)
save("$(fig_dir)/figSI_sim_beta_effect.png", fig)

fig
