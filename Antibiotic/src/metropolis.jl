# Import necessary libraries
import LinearAlgebra
import Distances

## =============================================================================
## Metropolis-Hastings Evolutionary Dynamics
## =============================================================================

## -----------------------------------------------------------------------------
## GaussianPeak struct
## -----------------------------------------------------------------------------

@doc raw"""
    GaussianPeak

A struct to hold the parameters of a single Gaussian peak.

# Fields
- `amplitude::AbstractFloat`: The amplitude of the peak.
- `mean::AbstractVector`: The mean of the peak.
- `covariance::AbstractMatrix`: The covariance matrix of the peak.
"""
struct GaussianPeak
    amplitude::AbstractFloat
    mean::AbstractVector
    covariance::AbstractMatrix
end

## -----------------------------------------------------------------------------
## Fitness function
## -----------------------------------------------------------------------------

@doc raw"""
    fitness(peak::GaussianPeak, x::AbstractVecOrMat; min_value::AbstractFloat=0.0)
    fitness(peaks::Vector{GaussianPeak}, x::AbstractVecOrMat; min_value::AbstractFloat=0.0)

Calculate the fitness value for a given phenotype `x` based on Gaussian peak(s).

# Arguments
- `x::AbstractVecOrMat`: The phenotype(s) for which to calculate the fitness.
  Can be a vector for a single phenotype or a matrix for multiple phenotypes,
  where each column corresponds to a phenotype.
- `peak::GaussianPeak`: A single Gaussian peak.
- `peaks::Vector{GaussianPeak}`: A vector of Gaussian peaks.
- `min_value::AbstractFloat=1.0`: The minimum fitness value to be added to the
  Gaussian contribution.

# Returns
The calculated fitness value(s).

# Description
The first method computes the fitness for a single Gaussian peak, while the
second method computes the sum of fitness values for multiple Gaussian peaks. In
both cases, the `min_value` is added to the Gaussian contribution to ensure a
minimum fitness level.
"""
function fitness(
    x::AbstractVector, peak::GaussianPeak; min_value::AbstractFloat=1.0
)
    # Compute the Square Mahalanobis distance
    sq_mahalanobis = Distances.evaluate(
        Distances.SqMahalanobis(
            LinearAlgebra.inv(peak.covariance),
            skipchecks=true
        ),
        x, peak.mean
    )
    # Calculate the Gaussian peak using broadcasting
    gaussian = peak.amplitude .* exp.(-0.5 .* sq_mahalanobis)
    # Return the fitness by shifting the Gaussian peak by the minimum value
    return gaussian + min_value
end

function fitness(
    x::AbstractMatrix, peak::GaussianPeak; min_value::AbstractFloat=1.0
)
    # Compute the Square Mahalanobis distance
    sq_mahalanobis = Distances.colwise(
        Distances.SqMahalanobis(
            LinearAlgebra.inv(peak.covariance),
            skipchecks=true
        ),
        x, peak.mean
    )
    # Calculate the Gaussian peak using broadcasting
    gaussian = peak.amplitude .* exp.(-0.5 .* sq_mahalanobis)
    # Return the fitness by shifting the Gaussian peak by the minimum value
    return gaussian .+ min_value
end

function fitness(
    x::AbstractVector, peaks::Vector{GaussianPeak}; min_value::AbstractFloat=1.0
)
    # Compute the Square Mahalanobis distance
    sq_mahalanobis = [
        Distances.evaluate(
            Distances.SqMahalanobis(
                LinearAlgebra.inv(peak.covariance),
                skipchecks=true
            ),
            x, peak.mean
        ) for peak in peaks
    ]
    # Sum the Gaussian contributions from all peaks using broadcasting
    total_gaussian = sum(
        peak.amplitude .* exp.(-0.5 .* sq_mahalanobis[i])
        for (i, peak) in enumerate(peaks)
    )

    # Return the fitness by shifting the Gaussian peak by the minimum value
    return total_gaussian + min_value
end

function fitness(
    x::AbstractMatrix, peaks::Vector{GaussianPeak}; min_value::AbstractFloat=1.0
)
    # Compute the Square Mahalanobis distance
    sq_mahalanobis = [
        Distances.colwise(
            Distances.SqMahalanobis(
                LinearAlgebra.inv(peak.covariance),
                skipchecks=true
            ),
            x, peak.mean
        ) for peak in peaks
    ]
    # Calculate the total Gaussian peak contribution using broadcasting
    total_gaussian = reduce(+, [
        peak.amplitude .* exp.(-0.5 .* sq_mahalanobis[i])
        for (i, peak) in enumerate(peaks)
    ])
    return total_gaussian .+ min_value
end

## -----------------------------------------------------------------------------
## Mutational landscape
## -----------------------------------------------------------------------------

@doc raw"""
    mutational_landscape(
        x::AbstractVecOrMat,
        peak::Union{GaussianPeak, Vector{GaussianPeak}};
        max_value::Float64 = 1.0
    )

Calculate the mutational landscape value for a given phenotype `x` based on
Gaussian peak(s).

# Arguments
- `peak::Union{GaussianPeak, Vector{GaussianPeak}}`: A single Gaussian peak or a
  vector of Gaussian peaks.
- `x::AbstractVecOrMat`: The phenotype(s) for which to calculate the mutational
  landscape. Can be a vector for a single phenotype or a matrix for multiple
  phenotypes, where each column corresponds to a phenotype.
- `max_value::Float64`: Optional. The maximum value of the landscape. Default is
  1.0.

# Returns
The calculated mutational landscape value(s).
"""
function mutational_landscape(
    x::AbstractVector, peak::GaussianPeak; max_value::AbstractFloat=1.0
)
    # Compute the Square Mahalanobis distance
    sq_mahalanobis = Distances.evaluate(
        Distances.SqMahalanobis(
            LinearAlgebra.inv(peak.covariance),
            skipchecks=true
        ),
        x, peak.mean
    )
    # Calculate the Gaussian peak
    gaussian = peak.amplitude * exp(-0.5 * sq_mahalanobis)
    # Return the mutational landscape by inverting the Gaussian peak shifted by
    # the maximum value
    return -(gaussian - max_value)
end

function mutational_landscape(
    x::AbstractMatrix, peak::GaussianPeak; max_value::AbstractFloat=1.0
)
    # Compute the Square Mahalanobis distance
    sq_mahalanobis = Distances.colwise(
        Distances.SqMahalanobis(
            LinearAlgebra.inv(peak.covariance),
            skipchecks=true
        ),
        x, peak.mean
    )
    # Calculate the Gaussian peak
    gaussian = peak.amplitude .* exp.(-0.5 .* sq_mahalanobis)
    # Return the mutational landscape by inverting the Gaussian peak shifted by
    # the maximum value
    return -(gaussian .- max_value)
end

function mutational_landscape(
    x::AbstractVector, peaks::Vector{GaussianPeak}; max_value::AbstractFloat=1.0
)
    # Compute the Square Mahalanobis distance
    sq_mahalanobis = [
        Distances.evaluate(
            Distances.SqMahalanobis(
                LinearAlgebra.inv(peak.covariance),
                skipchecks=true
            ),
            x, peak.mean
        ) for peak in peaks
    ]
    # Sum the Gaussian contributions from all peaks using broadcasting
    total_gaussian = sum(
        peak.amplitude .* exp.(-0.5 .* sq_mahalanobis[i])
        for (i, peak) in enumerate(peaks)
    )
    # Return the mutational landscape by inverting the Gaussian peak shifted by
    # the maximum value
    return -(total_gaussian - max_value)
end

function mutational_landscape(
    peaks::Vector{GaussianPeak},
    x::AbstractMatrix;
    max_value::AbstractFloat=1.0
)
    # Compute the Square Mahalanobis distance
    sq_mahalanobis = [
        Distances.colwise(
            Distances.SqMahalanobis(
                LinearAlgebra.inv(peak.covariance),
                skipchecks=true
            ),
            x, peak.mean
        ) for peak in peaks
    ]
    # Calculate the total Gaussian peak contribution using broadcasting
    total_gaussian = reduce(+, [
        peak.amplitude .* exp.(-0.5 .* sq_mahalanobis[i])
        for (i, peak) in enumerate(peaks)
    ])
    # Return the mutational landscape by inverting the Gaussian peak shifted by
    # the maximum value
    return -(total_gaussian .- max_value)
end

## -----------------------------------------------------------------------------
## Metropolis-Hastings Evolutionary Dynamics
## -----------------------------------------------------------------------------

@doc raw"""
    evo_metropolis_hastings(x0, fitness_peaks, mut_peaks, β, µ, n_steps)

Perform evolutionary Metropolis-Hastings algorithm to simulate phenotypic
evolution.

# Arguments
- `x0::AbstractVecOrMat`: Initial phenotype vector or matrix.
- `fitness_peaks::Union{GaussianPeak, Vector{GaussianPeak}}`: Fitness landscape
  defined by one or more Gaussian peaks.
- `mut_peaks::Union{GaussianPeak, Vector{GaussianPeak}}`: Mutational landscape
  defined by one or more Gaussian peaks.
- `β::AbstractFloat`: Inverse temperature parameter controlling selection
  strength.
- `µ::AbstractFloat`: Mutation step size standard deviation.
- `n_steps::Int`: Number of steps to simulate.

# Returns
- `Matrix{Float64}`: Matrix of phenotypes, where each column represents a step
  in the simulation.

# Description
This function implements the Metropolis-Hastings algorithm to simulate
phenotypic evolution in a landscape defined by fitness and mutational
accessibility. It uses the following steps:

1. Initialize the phenotype trajectory with the given starting point.
2. For each step: 
   a. Propose a new phenotype by adding Gaussian noise. 
   b. Calculate the fitness and mutational landscape values for the new 
      phenotype.
   c. Compute the acceptance probability based on the ratio of new and current 
      values. 
   d. Accept or reject the proposed phenotype based on the acceptance 
      probability.
3. Return the complete phenotype trajectory.

The acceptance probability is calculated using the simplified form: 
P_accept = min(1, (F_E(x_new) * M(x_new)) / (F_E(x) * M(x)))

where F_E is the fitness function and M is the mutational landscape function.
"""
function evo_metropolis_hastings(
    x0::AbstractVector,
    fitness_peaks::Union{GaussianPeak,Vector{GaussianPeak}},
    mut_peaks::Union{GaussianPeak,Vector{GaussianPeak}},
    β::Real,
    µ::Real,
    n_steps::Int,
)
    # Initialize array to hold phenotypes
    x = Matrix{Float64}(undef, length(x0), n_steps + 1)
    # Set initial phenotype
    x[:, 1] = x0

    # Compute fitness and mutational landscape at initial phenotype
    fitness_val = fitness(x0, fitness_peaks)
    mut_val = mutational_landscape(x0, mut_peaks)

    # Loop over steps
    for t in 1:n_steps
        # Propose new phenotype
        x_new = x[:, t] + µ * randn(length(x0))

        # Calculate fitness and mutational landscape
        fitness_val_new = fitness(x_new, fitness_peaks)
        mut_val_new = mutational_landscape(x_new, mut_peaks)

        # Compute acceptance probability
        P_accept = min(
            1, (fitness_val_new * mut_val_new / (fitness_val * mut_val))^β
        )

        # Accept or reject proposal
        if rand() < P_accept
            # Accept proposal
            x[:, t+1] = x_new
            # Update fitness and mutational landscape
            fitness_val = fitness_val_new
            mut_val = mut_val_new
        else
            # Reject proposal
            x[:, t+1] = x[:, t]
        end
    end # for
    return x
end # function

