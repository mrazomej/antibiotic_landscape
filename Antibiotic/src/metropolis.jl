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
    return GaussianPeaks(SS.StructArray(peaks))
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
## Genetic density
## -----------------------------------------------------------------------------

@doc raw"""
    genetic_density(
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
function genetic_density(
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

function genetic_density(
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

function genetic_density(
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

function genetic_density(
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

function genetic_density(
    x::AbstractVector,
    y::AbstractVector,
    peaks::AbstractPeak;
    max_value::AbstractFloat=1.0
)
    return genetic_density.(
        [[x, y] for x in x, y in y], Ref(peaks); max_value=max_value
    )
end

function genetic_density(
    xs::Tuple{AbstractVector,Vararg{AbstractVector}},
    peaks::AbstractPeak,
    max_value::AbstractFloat=1.0
)
    # Create array of all combinations of coordinates
    coords = [[x...] for x in IterTools.product(xs...)]

    # Compute mutational landscape
    mut = genetic_density.(coords, Ref(peaks); max_value=max_value)

    # Return
    return DD.DimArray(
        mut,
        tuple([DD.Dim{Symbol("x$i")}(1:length(xs[i])) for i in 1:length(xs)]...)
    )
end

## -----------------------------------------------------------------------------
## Metropolis-Hastings Evolutionary Dynamics
## -----------------------------------------------------------------------------

@doc raw"""
    evo_metropolis_hastings(x0, fitness_peaks, gen_peaks, β, µ, n_steps)

Perform evolutionary Metropolis-Hastings algorithm to simulate phenotypic
evolution.

# Arguments
- `x0::AbstractVecOrMat`: Initial phenotype vector or matrix.
- `fitness_peaks::Union{GaussianPeak, Vector{GaussianPeak}}`: Fitness landscape
  defined by one or more Gaussian peaks.
- `gen_peaks::Union{GaussianPeak, Vector{GaussianPeak}}`: Genetic density
  landscape defined by one or more Gaussian peaks.
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
    gen_peaks::AbstractPeak,
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
    gen_val = genetic_density(x0, gen_peaks)

    # Loop over steps
    for t in 1:n_steps
        # Propose new phenotype
        x_new = x[:, t] + µ * randn(length(x0))

        # Calculate fitness and mutational landscape
        fitness_val_new = fitness(x_new, fitness_peaks)
        gen_val_new = genetic_density(x_new, gen_peaks)

        # Compute acceptance probability
        P_accept = min(
            1, (fitness_val_new * gen_val_new / (fitness_val * gen_val))^β
        )

        # Accept or reject proposal
        if rand() < P_accept
            # Accept proposal
            x[:, t+1] = x_new
            # Update fitness and mutational landscape
            fitness_val = fitness_val_new
            gen_val = gen_val_new
        else
            # Reject proposal
            x[:, t+1] = x[:, t]
        end
    end # for
    return x
end # function

@doc raw"""
    evo_metropolis_hastings(x0, fitness_peaks, gen_peaks, β, µ, n_steps, bounds)

Similar to the unbounded version, but with bounds on the phenotype space.

# Additional Arguments
- `bounds::Vector{Tuple{Float64, Float64}}`: Vector of (min, max) tuples
  specifying the allowed range for each dimension.
"""
function evo_metropolis_hastings(
    x0::AbstractVector,
    fitness_peaks::AbstractPeak,
    gen_peaks::AbstractPeak,
    β::Real,
    µ::Real,
    n_steps::Int,
    bounds::Vector{Tuple{Float64,Float64}}
)
    # Check that number of bounds matches dimensions
    if length(bounds) != length(x0)
        throw(ArgumentError("Number of bounds ($(length(bounds))) must match number of dimensions ($(length(x0)))"))
    end

    # Check that initial point is within bounds
    if !all(b[1] <= x <= b[2] for (x, b) in zip(x0, bounds))
        throw(ArgumentError("Initial point must be within bounds"))
    end

    # Initialize array to hold phenotypes
    x = Matrix{Float64}(undef, length(x0), n_steps + 1)
    # Set initial phenotype
    x[:, 1] = x0

    # Compute fitness and mutational landscape at initial phenotype
    fitness_val = fitness(x0, fitness_peaks)
    gen_val = genetic_density(x0, gen_peaks)

    # Loop over steps
    for t in 1:n_steps
        # Propose new phenotype
        x_new = x[:, t] + µ * randn(length(x0))

        # Check if proposal is within bounds
        within_bounds = all(b[1] <= x <= b[2] for (x, b) in zip(x_new, bounds))

        if within_bounds
            # Calculate fitness and mutational landscape
            fitness_val_new = fitness(x_new, fitness_peaks)
            gen_val_new = genetic_density(x_new, gen_peaks)

            # Compute acceptance probability
            P_accept = min(
                1, (fitness_val_new * gen_val_new / (fitness_val * gen_val))^β
            )

            # Accept or reject proposal
            if rand() < P_accept
                # Accept proposal
                x[:, t+1] = x_new
                # Update fitness and mutational landscape
                fitness_val = fitness_val_new
                gen_val = gen_val_new
            else
                # Reject proposal
                x[:, t+1] = x[:, t]
            end
        else
            # Automatically reject out-of-bounds proposals
            x[:, t+1] = x[:, t]
        end
    end
    return x
end

## -----------------------------------------------------------------------------
# Metropolis-Kimura evolutionary dynamics
## -----------------------------------------------------------------------------

@doc raw"""
    evo_metropolis_kimura(x0, fitness_peaks, gen_peaks, N, µ, n_steps)

Perform evolutionary algorithm to simulate phenotypic evolution using a
Metropolis-Hastring-like mutation probability and Kimura's fixation probability.

# Arguments
- `x0::AbstractVector`: Initial phenotype vector.
- `fitness_peaks::AbstractPeak`: Fitness landscape defined by one or more
  Gaussian peaks.
- `gen_peaks::AbstractPeak`: Genetic density landscape defined by one or more
  Gaussian peaks.
- `N::Real`: Effective population size.
- `µ::Real`: Mutation step size standard deviation.
- `n_steps::Int`: Number of steps to simulate.

# Returns
- `Matrix{Float64}`: Matrix of phenotypes, where each column represents a step
  in the simulation.

# Description
This function implements the Metropolis-Hastings algorithm to simulate
phenotypic evolution in a landscape defined by fitness and mutational
accessibility. It uses Kimura's fixation probability formula to determine
acceptance of proposed mutations. The algorithm follows these steps:

1. Initialize the phenotype trajectory with the given starting point.
2. For each step: a. Propose a new phenotype by adding Gaussian noise. b.
   Calculate the fitness and mutational landscape values for the new phenotype.
   c. Compute the selection coefficient s = (F_new - F)/F d. Calculate mutation
      probability P_mut = min(1, M_new/M) e. Calculate fixation probability
   using Kimura's equation: P_fix = (1 - exp(-2s))/(1 - exp(-2Ns)) f. Accept or
   reject based on P_accept = P_mut * P_fix
3. Return the complete phenotype trajectory.

where F is the fitness function and M is the mutational landscape function.
"""
function evo_metropolis_kimura(
    x0::AbstractVector,
    fitness_peaks::AbstractPeak,
    gen_peaks::AbstractPeak,
    N::Real,
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
    gen_val = genetic_density(x0, gen_peaks)

    # Loop over steps
    for t in 1:n_steps
        # Propose new phenotype
        x_new = x[:, t] + µ * randn(length(x0))

        # Calculate fitness and mutational landscape
        fitness_val_new = fitness(x_new, fitness_peaks)
        gen_val_new = genetic_density(x_new, gen_peaks)

        # Compute selection coefficient
        s = (fitness_val_new - fitness_val) /
            (fitness_val + eps(eltype(fitness_val)))

        # Compute mutation probability
        P_mut = min(1, (gen_val_new / gen_val)^β)

        # Compute fixation probability using Kimura's equation
        # Use Taylor expansion approximation when s << N for numerical stability
        if (1 / (2 * N)) * 10 < s < 0.1
            # For small s, use Taylor expansion
            P_fix = 2 * s
        else
            # Otherwise use full Kimura equation
            P_fix = (1 - exp(-2 * s)) / (1 - exp(-2 * N * s))
        end

        # Compute acceptance probability
        P_accept = P_mut * P_fix

        # Accept or reject proposal
        if rand() < P_accept
            # Accept proposal
            x[:, t+1] = x_new
            # Update fitness and mutational landscape
            fitness_val = fitness_val_new
            gen_val = gen_val_new
        else
            # Reject proposal
            x[:, t+1] = x[:, t]
        end
    end # for
    return x
end # function

## -----------------------------------------------------------------------------

@doc raw"""
    evo_metropolis_kimura(x0, fitness_peaks, gen_peaks, N, µ, n_steps, bounds)

Similar to the unbounded version, but with bounds on the phenotype space.

# Additional Arguments
- `bounds::Vector{Tuple{Float64, Float64}}`: Vector of (min, max) tuples
  specifying the allowed range for each dimension.
"""
function evo_metropolis_kimura(
    x0::AbstractVector,
    fitness_peaks::AbstractPeak,
    gen_peaks::AbstractPeak,
    N::Real,
    µ::Real,
    n_steps::Int,
    bounds::Vector{Tuple{Float64,Float64}}
)
    # Check that number of bounds matches dimensions
    if length(bounds) != length(x0)
        throw(ArgumentError("Number of bounds ($(length(bounds))) must match number of dimensions ($(length(x0)))"))
    end

    # Check that initial point is within bounds
    if !all(b[1] <= x <= b[2] for (x, b) in zip(x0, bounds))
        throw(ArgumentError("Initial point must be within bounds"))
    end

    # Initialize array to hold phenotypes
    x = Matrix{Float64}(undef, length(x0), n_steps + 1)
    # Set initial phenotype
    x[:, 1] = x0

    # Compute fitness and mutational landscape at initial phenotype
    fitness_val = fitness(x0, fitness_peaks)
    gen_val = genetic_density(x0, gen_peaks)

    # Loop over steps
    for t in 1:n_steps
        # Propose new phenotype
        x_new = x[:, t] + µ * randn(length(x0))

        # Check if proposal is within bounds
        within_bounds = all(b[1] <= x <= b[2] for (x, b) in zip(x_new, bounds))

        if within_bounds
            # Calculate fitness and mutational landscape
            fitness_val_new = fitness(x_new, fitness_peaks)
            gen_val_new = genetic_density(x_new, gen_peaks)

            # Compute selection coefficient
            s = (fitness_val_new - fitness_val) / fitness_val

            # Compute mutation probability
            P_mut = min(
                1, gen_val_new / gen_val
            )

            # Compute fixation probability using Kimura's equation
            P_fix = (1 - exp(-2 * s)) / (1 - exp(-2 * N * s))

            # Compute acceptance probability
            P_accept = P_mut * P_fix

            # Accept or reject proposal
            if rand() < P_accept
                # Accept proposal
                x[:, t+1] = x_new
                # Update fitness and mutational landscape
                fitness_val = fitness_val_new
                gen_val = gen_val_new
            else
                # Reject proposal
                x[:, t+1] = x[:, t]
            end
        else
            # Automatically reject out-of-bounds proposals
            x[:, t+1] = x[:, t]
        end
    end
    return x
end