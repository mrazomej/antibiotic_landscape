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
n_dim = 2
# Number of simulations (evolving strains)
n_sim = 10
# Inverse temperature
β = 10.0
# mutation step size
µ = 0.1
# Number of evolution steps
n_steps = 300

# Define number of fitness landscapes
n_fit_lans = 50

# Define range of peak means
peak_mean_min = -4.0
peak_mean_max = 4.0

# Define range of fitness amplitudes
fit_amp_min = 1.0
fit_amp_max = 5.0

# Define covariance range
fit_cov_min = 0.5
fit_cov_max = 3.0

# Define possible number of fitness peaks
n_fit_peaks_min = 1
n_fit_peaks_max = 3

## =============================================================================

println("Defining SPECIFIC evolution condition...")

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

## =============================================================================

Random.seed!(42)

println("Simulating evolution...")

# Select initial conditions relatively close to each other
x0 = rand(Distributions.MvNormal([-2.5, -2.5], 0.1), n_sim)

# Define dimensions to be used with DimensionalData
phenotype = DD.Dim{:phenotype}([:x1, :x2])
fitness = DD.Dim{:fitness}([:fitness])
time = DD.Dim{:time}(0:n_steps)
lineage = DD.Dim{:lineage}(1:n_sim)


# Initialize DimensionalData array to hold trajectories and fitness
phenotype_traj = DD.zeros(Float32, phenotype, time, lineage)
fitness_traj = DD.zeros(Float32, fitness, time, lineage)

# Stack arrays to store trajectories in phenotype and fitness dimensions
x_traj = DD.DimStack(
    (phenotype=phenotype_traj, fitness=fitness_traj),
)

# Store initial conditions
x_traj.phenotype[time=1] = x0
x_traj.fitness[time=1] = mh.fitness(x0, fit_evo_peak)

# Loop over simulations
for i in 1:n_sim
    # Run Metropolis-Hastings algorithm
    trajectory = mh.evo_metropolis_hastings(
        x_traj.phenotype[time=1, lineage=i],
        fit_evo_peak,
        mut_evo_peaks,
        β,
        µ,
        n_steps
    )

    # Store trajectory
    x_traj.phenotype[lineage=i] = trajectory

    # Calculate and store fitness for each point in the trajectory
    x_traj.fitness[lineage=i] = mh.fitness(trajectory, fit_evo_peak)
end

## =============================================================================

Random.seed!(42)

println("Generating alternative fitness landscapes...")

# Initialize array to hold fitness landscapes
fit_lans = Array{mh.AbstractPeak}(undef, n_fit_lans + 1)

# Store evolution condition in first landscape
fit_lans[1] = fit_evo_peak

# Loop over number of fitness landscapes
for i in 1:n_fit_lans
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
        fit_lans[i+1] = mh.GaussianPeak(
            first(fit_amplitudes), first(fit_means), first(fit_covariances)
        )
    else
        # Create fitness peaks
        fit_lans[i+1] = mh.GaussianPeaks(
            fit_amplitudes, fit_means, fit_covariances
        )
    end # if
end # for

## =============================================================================

println("Computing fitnotype profiles...")

# Define landscape dimension
landscape = DD.Dim{:landscape}(1:n_fit_lans+1)

# Initialize fitness and phenotype profiles
fitness_profiles = DD.zeros(Float32, landscape, time, lineage)
phenotype_profiles = DD.zeros(Float32, phenotype, time, lineage)

# Initialize DimensionalData array to hold fitnotype profiles
fitnotype_profiles = DD.DimStack(
    (phenotype=phenotype_profiles, fitness=fitness_profiles),
)

# Store evolution condition in first landscape
fitnotype_profiles.phenotype .= x_traj.phenotype
fitnotype_profiles.fitness[landscape=1] = x_traj.fitness

# Loop over fitness landscapes
for lan in DD.dims(fitnotype_profiles, :landscape)[2:end]
    # Loop through lineages
    for lin in DD.dims(fitnotype_profiles, :lineage)
        # Store fitness trajectories
        fitnotype_profiles.fitness[landscape=lan, lineage=lin] = mh.fitness(
            fitnotype_profiles.phenotype[lineage=lin].data,
            fit_lans[lan]
        )
    end # for
end # for

## =============================================================================

println("Saving results...")

# Save results
JLD2.jldsave(
    "$(sim_dir)/sim_evo.jld2",
    fitnotype_profiles=fitnotype_profiles,
    fitness_landscapes=fit_lans,
    mutational_landscape=mut_evo_peaks,
    evolution_condition=fit_evo_peak,
)
