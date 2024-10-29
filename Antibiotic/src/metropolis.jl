# Import necessary libraries
import LinearAlgebra
import Distances
import StructArrays as SS
import DimensionalData as DD
import IterTools
using ConcreteStructs: @concrete

## =============================================================================
## Metropolis-Hastings Evolutionary Dynamics
## =============================================================================

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## GaussianPeak struct
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

@doc raw"""
    AbstractPeak

An abstract type for representing a peak in a fitness or mutational landscape.
"""
abstract type AbstractPeak end

@doc raw"""
    GaussianPeak

A struct to hold the parameters of a single Gaussian peak.

# Fields
- `amplitude::AbstractFloat`: The amplitude of the peak.
- `mean::AbstractVector`: The mean of the peak.
- `covariance::AbstractMatrix`: The covariance matrix of the peak.
"""
@concrete struct GaussianPeak <: AbstractPeak
    amplitude::AbstractFloat
    mean::AbstractVector
    covariance::AbstractMatrix
end

@doc raw"""
    GaussianPeaks

A struct to hold multiple Gaussian peaks.

# Fields
- `peaks::Vector{GaussianPeak}`: A vector of GaussianPeak objects.
"""
@concrete struct GaussianPeaks <: AbstractPeak
    peaks::SS.StructArray{GaussianPeak}
end

## -----------------------------------------------------------------------------
## Constructors
## -----------------------------------------------------------------------------

# Constructor for GaussianPeak for diagonal covariance matrices
function GaussianPeak(
    amplitude::AbstractFloat, mean::AbstractVector, variance::AbstractFloat
)
    return GaussianPeak(
        amplitude,
        mean,
        LinearAlgebra.Diagonal(repeat([variance], length(mean)))
    )
end

## -----------------------------------------------------------------------------

# Constructor for GaussianPeaks from separate arrays
function GaussianPeaks(
    amplitudes::AbstractVector,
    means::AbstractMatrix,
    covariances::AbstractArray
)
    # Check that dimensions match
    @assert size(means, 2) == length(amplitudes) == size(covariances, 3) "Dimensions must match"
    # Create vector of GaussianPeak objects
    peaks = [
        GaussianPeak(amplitudes[i], means[:, i], covariances[:, :, i])
        for i in 1:length(amplitudes)
    ]
    # Return GaussianPeaks object
    return GaussianPeaks(peaks)
end

## -----------------------------------------------------------------------------

# Constructor for GaussianPeaks with equal amplitude and equal diagonal covariance
function GaussianPeaks(
    amplitude::AbstractFloat,
    mean::AbstractVector{<:AbstractVector},
    variance::AbstractFloat
)
    return GaussianPeaks(
        SS.StructArray([
            GaussianPeak(amplitude, mean[i], variance) for i in 1:length(mean)
        ])
    )
end

## -----------------------------------------------------------------------------

# Constructor for GaussianPeaks with diagonal covariance
function GaussianPeaks(
    amplitude::AbstractVector{<:AbstractFloat},
    mean::AbstractVector{<:AbstractVector},
    variance::AbstractVector{<:AbstractFloat}
)
    @assert length(amplitude) == length(mean) == length(variance) "Lengths must match"
    return GaussianPeaks(
        SS.StructArray([
            GaussianPeak(amplitude[i], mean[i], variance[i])
            for i in 1:length(amplitude)
        ])
    )
end

## -----------------------------------------------------------------------------
## Size methods
## -----------------------------------------------------------------------------

function Base.length(peaks::GaussianPeaks)
    return length(peaks.peaks)
end

function Base.length(peak::GaussianPeak)
    return 1
end

function Base.size(peaks::GaussianPeaks)
    return (dim=length(peaks.peaks[1].mean), n_peaks=length(peaks.peaks))
end

function Base.size(peak::GaussianPeak)
    return (dim=length(peak.mean), n_peaks=1)
end

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Fitness function
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

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
    x::AbstractVector, peaks::GaussianPeaks; min_value::AbstractFloat=1.0
)
    # Compute the Square Mahalanobis distance
    sq_mahalanobis = [
        Distances.evaluate(
            Distances.SqMahalanobis(
                LinearAlgebra.inv(peak.covariance),
                skipchecks=true
            ),
            x, peak.mean
        ) for peak in peaks.peaks
    ]
    # Sum the Gaussian contributions from all peaks using broadcasting
    total_gaussian = sum(
        peak.amplitude .* exp.(-0.5 .* sq_mahalanobis[i])
        for (i, peak) in enumerate(peaks.peaks)
    )

    # Return the fitness by shifting the Gaussian peak by the minimum value
    return total_gaussian + min_value
end

function fitness(
    x::AbstractMatrix, peaks::GaussianPeaks; min_value::AbstractFloat=1.0
)
    # Compute the Square Mahalanobis distance
    sq_mahalanobis = [
        Distances.colwise(
            Distances.SqMahalanobis(
                LinearAlgebra.inv(peak.covariance),
                skipchecks=true
            ),
            x, peak.mean
        ) for peak in peaks.peaks
    ]
    # Calculate the total Gaussian peak contribution using broadcasting
    total_gaussian = reduce(+, [
        peak.amplitude .* exp.(-0.5 .* sq_mahalanobis[i])
        for (i, peak) in enumerate(peaks.peaks)
    ])
    return total_gaussian .+ min_value
end

function fitness(
    x::AbstractArray{T,3} where {T},
    peaks::AbstractPeak;
    min_value::AbstractFloat=1.0
)
    # Get the size of the 3D tensor
    s1, s2, s3 = size(x)

    # Preallocate the result array
    result = Array{eltype(x)}(undef, s2, s3)

    # Apply the AbstractMatrix method to each slice
    for i in 1:s3
        result[:, i] = fitness(view(x, :, :, i), peaks; min_value=min_value)
    end

    return result
end

function fitness(
    x::AbstractArray,
    peaks::Vector{<:AbstractPeak};
    min_value::AbstractFloat=1.0
)
    # Apply fitness function to each peak and collect results
    results = [fitness(x, peak; min_value=min_value) for peak in peaks]

    # Concatenate results along a new dimension
    return cat(results..., dims=ndims(x) + 1)
end

function fitness(
    x::AbstractVector,
    y::AbstractVector,
    peaks::AbstractPeak;
    min_value::AbstractFloat=1.0
)
    return fitness.(
        [[x, y] for x in x, y in y], Ref(peaks); min_value=min_value
    )
end

function fitness(
    xs::Tuple{AbstractVector,Vararg{AbstractVector}},
    peaks::AbstractPeak,
    min_value::AbstractFloat=1.0
)
    # Create array of all combinations of coordinates
    coords = [[x...] for x in IterTools.product(xs...)]

    # Compute fitness
    fit = fitness.(coords, Ref(peaks); min_value=min_value)

    # Return
    return DD.DimArray(
        fit,
        tuple([DD.Dim{Symbol("x$i")}(1:length(xs[i])) for i in 1:length(xs)]...)
    )
end

## -----------------------------------------------------------------------------
## Mutational landscape
## -----------------------------------------------------------------------------

@doc raw"""
    mutational_landscape(
        x::AbstractVecOrMat,
        peak::Union{GaussianPeak, GaussianPeaks};
        max_value::Float64 = 1.0
    )

Calculate the mutational landscape value for a given phenotype `x` based on
Gaussian peak(s).

# Arguments
- `peak::Union{GaussianPeak, GaussianPeaks}`: A single Gaussian peak or a
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
    x::AbstractVector, peaks::GaussianPeaks; max_value::AbstractFloat=1.0
)
    # Compute the Square Mahalanobis distance
    sq_mahalanobis = [
        Distances.evaluate(
            Distances.SqMahalanobis(
                LinearAlgebra.inv(peak.covariance),
                skipchecks=true
            ),
            x, peak.mean
        ) for peak in peaks.peaks
    ]
    # Sum the Gaussian contributions from all peaks using broadcasting
    total_gaussian = sum(
        peak.amplitude .* exp.(-0.5 .* sq_mahalanobis[i])
        for (i, peak) in enumerate(peaks.peaks)
    )
    # Return the mutational landscape by inverting the Gaussian peak shifted by
    # the maximum value
    return -(total_gaussian - max_value)
end

function mutational_landscape(
    peaks::GaussianPeaks,
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
        ) for peak in peaks.peaks
    ]
    # Calculate the total Gaussian peak contribution using broadcasting
    total_gaussian = reduce(+, [
        peak.amplitude .* exp.(-0.5 .* sq_mahalanobis[i])
        for (i, peak) in enumerate(peaks.peaks)
    ])
    # Return the mutational landscape by inverting the Gaussian peak shifted by
    # the maximum value
    return -(total_gaussian .- max_value)
end

function mutational_landscape(
    x::AbstractVector,
    y::AbstractVector,
    peaks::AbstractPeak;
    max_value::AbstractFloat=1.0
)
    return mutational_landscape.(
        [[x, y] for x in x, y in y], Ref(peaks); max_value=max_value
    )
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
    fitness_peaks::AbstractPeak,
    mut_peaks::AbstractPeak,
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

