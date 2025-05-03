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

# Import IterTools for iterating over Cartesian products
import IterTools

# Import basic math libraries
import StatsBase
import LinearAlgebra
import Random
import Distributions
import Distances

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

# Define output directory
out_dir = "$(git_root())/output$(out_prefix)"

# Generate output directory if it doesn't exist
if !isdir(out_dir)
    println("Generating output directory...")
    mkpath(out_dir)
end

# Define simulation directory
sim_dir = "$(out_dir)/sim_evo"

# Generate simulation directory if it doesn't exist
if !isdir(sim_dir)
    println("Generating simulation directory...")
    mkpath(sim_dir)
end

## =============================================================================

println("Defining evolution parameters...")

# Phenotype space dimensionality
n_dim = 2
# Number of initial conditions (positions in phenotype space)
n_sim = 50
# Number of replicates (evolving strains per initial condition)
n_rep = 2
# Effective population size
N = 10^3
# Inverse temperature
β = 10.0
# mutation step size
µ = 0.1
# Number of evolution steps
n_steps = 3000
# Define number of subsampling steps
sub_rate = 10

# Define number of fitness landscapes
n_fit_lans = 50

# Define range of peak means
peak_mean_min = -4.0
peak_mean_max = 4.0

# Define range of fitness amplitudes
fit_amp_min = 1.0
fit_amp_max = 5.0

# Define covariance range
fit_cov_min = 3.0
fit_cov_max = 10.0

# Define possible number of fitness peaks
n_fit_peaks_min = 1
n_fit_peaks_max = 4

## =============================================================================

println("Defining SPECIFIC mutational landscape...")

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

println("Generating alternative fitness landscapes...")

# Initialize array to hold fitness landscapes
fit_lans = Array{mh.AbstractPeak}(undef, n_fit_lans)

# Add fixed evolution condition to fitness landscapes
fit_lans[1] = fit_evo_peak

# Loop over number of fitness landscapes
for i in 2:n_fit_lans
    # Sample number of fitness peaks
    n_fit_peaks = rand(n_fit_peaks_min:n_fit_peaks_max)

    # Sample fitness means as 2D vectors from uniform distribution
    fit_means = [
        rand(Distributions.Uniform(peak_mean_min, peak_mean_max), 2)
        for _ in 1:n_fit_peaks
    ]

    # Sample fitness amplitudes from uniform distribution
    fit_amplitudes = rand(
        Distributions.Uniform(fit_amp_min, fit_amp_max), n_fit_peaks
    )

    # Sample fitness covariances from uniform distribution
    fit_covariances = rand(
        Distributions.Uniform(fit_cov_min, fit_cov_max), n_fit_peaks
    )

    # Check dimensionality
    if n_fit_peaks == 1
        # Create fitness peaks
        fit_lans[i] = mh.GaussianPeak(
            first(fit_amplitudes), first(fit_means), first(fit_covariances)
        )
    else
        # Create fitness peaks
        fit_lans[i] = mh.GaussianPeaks(
            fit_amplitudes, fit_means, fit_covariances
        )
    end # if
end # for

## =============================================================================

Random.seed!(42)

println("Simulating evolution and computing fitnotype profiles...")

# Sample initial positions on phenotype space from uniform distribution taking
# into account mutational landscape (not to start at a low mutational peak)

# Normalize mut_evo_grid_points
mut_evo_grid_points = (mut_evo_grid_points .- minimum(mut_evo_grid_points)) ./
                      (maximum(mut_evo_grid_points) - minimum(mut_evo_grid_points))

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
time = DD.Dim{:time}(0:sub_rate:n_steps) # time
lineage = DD.Dim{:lineage}(1:n_sim) # lineage
replicate = DD.Dim{:replicate}(1:n_rep) # replicate
landscape = DD.Dim{:landscape}(1:n_fit_lans) # landscape
evo = DD.Dim{:evo}(1:n_fit_lans) # evolution condition


# Initialize DimensionalData array to hold trajectories and fitness
phenotype_traj = DD.zeros(
    Float64,
    phenotype,
    time,
    lineage,
    replicate,
    landscape,
    evo,
)
fitness_traj = DD.zeros(
    Float64,
    fitness,
    time,
    lineage,
    replicate,
    landscape,
    evo,
)


# Stack arrays to store trajectories in phenotype and fitness dimensions
x_traj = DD.DimStack(
    (phenotype=phenotype_traj, fitness=fitness_traj),
)

# Store initial conditions
x_traj.phenotype[time=1] = repeat(
    x0_reps,
    outer=(1, 1, 1, n_fit_lans, n_fit_lans)
)

# Map initial phenotype to fitness
x_traj.fitness[time=1] = repeat(
    reduce(
        (x, y) -> cat(x, y, dims=3),
        mh.fitness.(Ref(x0_reps), fit_lans)
    ),
    outer=(1, 1, 1, 1, n_fit_lans)
)

# Loop over landscapes
Threads.@threads for evo in DD.dims(x_traj, :evo)
    # Loop over lineages
    for lin in DD.dims(x_traj, :lineage)
        # Loop over replicates
        for rep in DD.dims(x_traj, :replicate)
            # Run Metropolis-Hastings algorithm
            trajectory = mh.evo_metropolis_kimura(
                x_traj.phenotype[
                    time=1,
                    lineage=lin,
                    replicate=rep,
                    landscape=evo,
                    evo=evo,
                ].data,
                fit_lans[evo],
                mut_evo_peaks,
                N,
                β,
                µ,
                n_steps
            )

            # Store trajectory
            x_traj.phenotype[
                lineage=lin,
                replicate=rep,
                evo=evo,
            ] .= trajectory[:, 1:sub_rate:end]

            # Calculate and store fitness for each point in the trajectory
            x_traj.fitness[
                lineage=lin,
                replicate=rep,
                evo=evo,
            ] = mh.fitness(trajectory[:, 1:sub_rate:end], fit_lans)
        end # for
    end # for
end # for

## =============================================================================

println("Saving results...")

# Save results
JLD2.jldsave(
    "$(sim_dir)/sim_evo.jld2",
    fitnotype_profiles=x_traj,
    fitness_landscapes=fit_lans,
    genetic_density=mut_evo_peaks,
)