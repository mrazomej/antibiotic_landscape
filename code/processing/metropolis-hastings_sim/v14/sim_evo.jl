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
n_dim = 3
# Number of initial conditions (positions in phenotype space)
n_sim = 10
# Number of replicates (evolving strains per initial condition)
n_rep = 2
# Inverse temperature
β = 100.0
# mutation step size
µ = 0.1
# Number of evolution steps
n_steps = 300

# Define number of fitness landscapes
n_fit_lans = 25

# Define range of peak means
peak_mean_min = -4.0
peak_mean_max = 4.0

# Define range of fitness amplitudes
fit_amp_min = 1.0
fit_amp_max = 5.0

# Define covariance range
fit_cov_min = 1.5
fit_cov_max = 5.0

# Define possible number of fitness peaks
n_fit_peaks_min = 1
n_fit_peaks_max = 4

## =============================================================================

println("Defining SPECIFIC mutational landscape...")

# Mutational peak amplitude
mut_evo_amplitude = 1.0
# Mutational peak means
mut_means = [
    [-1.5, -1.5, -1.5],  # bottom left back
    [1.5, -1.5, -1.5],   # bottom right back 
    [1.5, 1.5, -1.5],    # top right back
    [-1.5, 1.5, -1.5],   # top left back
    [-1.5, -1.5, 1.5],   # bottom left front
    [1.5, -1.5, 1.5],    # bottom right front
    [1.5, 1.5, 1.5],     # top right front
    [-1.5, 1.5, 1.5],    # top left front
]
# Mutational peak covariance
mut_evo_covariance = 0.45
# Create mutational peaks
mut_evo_peaks = mh.GaussianPeaks(
    mut_evo_amplitude,
    mut_means,
    mut_evo_covariance
)

## =============================================================================

println("Defining FIXED evolution condition...")

# Evolution condition amplitude
fit_evo_amplitude = 5.0
# Evolution condition mean
fit_evo_mean = [0.0, 0.0, 0.0]
# Evolution condition covariance
fit_evo_covariance = [3.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 300.0]
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

# Initialize array to hold ignored dimensions
ignored_dims = Array{Int}(undef, n_fit_lans)

# Add initial ignored dimension
ignored_dims[1] = 3

# Loop over number of fitness landscapes
for i in 2:n_fit_lans
    # Sample dimension that will be ignored
    ignored_dims[i] = rand(1:3)

    # Sample number of fitness peaks
    n_fit_peaks = rand(n_fit_peaks_min:n_fit_peaks_max)

    # Sample fitness means as 3D vectors from uniform distribution
    fit_means = reduce(
        hcat,
        [
            rand(Distributions.Uniform(peak_mean_min, peak_mean_max), n_dim)
            for _ in 1:n_fit_peaks
        ]
    )

    # Sample fitness amplitudes from uniform distribution
    fit_amplitudes = rand(
        Distributions.Uniform(fit_amp_min, fit_amp_max), n_fit_peaks
    )

    # Sample fitness covariances from uniform distribution for non-ignored
    # dimensions
    fit_covariance_values = rand(
        Distributions.Uniform(fit_cov_min, fit_cov_max), n_fit_peaks
    )

    # Create covariance matrix for each peak
    fit_covariances = reduce(
        (x, y) -> cat(x, y, dims=3),
        [
            begin
                # Initialize 3x3 zero matrix
                cov_matrix = zeros(3, 3)
                # Fill diagonal with regular covariance except for ignored
                # dimension
                for j in 1:3
                    cov_matrix[j, j] = j == ignored_dims[i] ? 100.0 : fit_covariance_value
                end
                cov_matrix
            end for fit_covariance_value in fit_covariance_values
        ]
    )

    # Check dimensionality
    if n_fit_peaks == 1
        # Create fitness peaks
        fit_lans[i] = mh.GaussianPeak(
            first(fit_amplitudes),
            fit_means[:],
            fit_covariances
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

# Sample initial positions on phenotype space from uniform distribution
# between peak bounds
x0 = rand(Distributions.Uniform(peak_mean_min, peak_mean_max), (n_dim, n_sim))

# Select initial conditions for replicates from multivariate normal distribution
# around each initial condition
x0_reps = reduce(
    (x, y) -> cat(x, y, dims=3),
    [
        rand(Distributions.MvNormal(x0[:, i], 0.1), n_rep)
        for i in 1:n_sim
    ]
)
# Change order of dimensions
x0_reps = permutedims(x0_reps, (1, 3, 2))

# Define dimensions to be used with DimensionalData
phenotype = DD.Dim{:phenotype}([Symbol("x$i") for i in 1:n_dim]) # phenotype
fitness = DD.Dim{:fitness}([:fitness]) # fitness
time = DD.Dim{:time}(0:n_steps) # time
lineage = DD.Dim{:lineage}(1:n_sim) # lineage
replicate = DD.Dim{:replicate}(1:n_rep) # replicate
landscape = DD.Dim{:landscape}(1:n_fit_lans) # landscape
evo = DD.Dim{:evo}(1:n_fit_lans) # evolution condition

# Initialize DimensionalData array to hold trajectories and fitness
phenotype_traj = DD.zeros(
    Float32,
    phenotype,
    time,
    lineage,
    replicate,
    landscape,
    evo,
)
fitness_traj = DD.zeros(
    Float32,
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
for evo in DD.dims(x_traj, :evo)
    # Loop over lineages
    for lin in DD.dims(x_traj, :lineage)
        # Loop over replicates
        for rep in DD.dims(x_traj, :replicate)
            # Run Metropolis-Hastings algorithm
            trajectory = mh.evo_metropolis_hastings(
                x_traj.phenotype[
                    time=1,
                    lineage=lin,
                    replicate=rep,
                    landscape=evo,
                    evo=evo,
                ].data,
                fit_lans[evo],
                mut_evo_peaks,
                β,
                µ,
                n_steps
            )

            # Store trajectory
            x_traj.phenotype[
                lineage=lin,
                replicate=rep,
                evo=evo,
            ] .= trajectory

            # Calculate and store fitness for each point in the trajectory
            x_traj.fitness[
                lineage=lin,
                replicate=rep,
                evo=evo,
            ] = mh.fitness(trajectory, fit_lans)
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
    ignored_dimensions=ignored_dims,
    genetic_density=mut_evo_peaks,
)
