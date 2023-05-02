# Import basic math
import LinearAlgebra

# Import library to perform Einstein summation
using TensorOperations: @tensor

# Import library for automatic differentiation
import Zygote

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Differential Geometry on Riemmanian Manifolds
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    riemmanian_metric(manifold, val)

Function to compute the metric `M̲̲ = J̲̲ᵀJ̲̲` of a Riemmanian manifold via
numerical differentiation with `Zygote.jl`.

# Arguments
- `manifold::Function`: Function defining the manifold.
- `val::Vector{<:AbstractFloat}`: Value where to evaluate the metric.

# Returns
- `M̲̲::Matrix{<:AbstractFloat}`: Matrix evaluating the Riemmanian manifold
  metric.
"""
function riemmanian_metric(
    manifold::Function, val::Vector{T}
)::Array{T} where {T<:AbstractFloat}
    # Compute Jacobian
    jac = first(Zygote.jacobian(manifold, val))
    # Initialize
    return jac' * jac
end # function

@doc raw"""
    ∂M̲̲∂γ̲(manifold, val, out_dim)

Function to compute the derivative of the Riemmanian metric `M̲̲` as a function
of the coordinates in the input space using `Zygote.jl`.

# Arguments
- `manifold::Function`: Function defining the manifold.
- `val::Vector{<:AbstractFloat}`: Value where to evaluate the metric.
- `out_dim::Int`: Dimensionality of the output space. This is required because
  `Zygote.jl` cannot compute the required derivatives automatically, but we have
  to "manually" iterate over the variables in the output space, one at the time.

# Returns
- `∂M̲̲::Array{<:AbstractFloat}`: Rank-3 tensor evaluating the derivative of the
  Riemmanian manifold metric.
"""
function ∂M̲̲∂γ̲(
    manifold::Function, val::Vector{T}, out_dim::Int
)::Array{T} where {T<:AbstractFloat}
    # Compute the manifold Jacobian. This should be a D×d matrix where D is the
    # dimension of the output space, and d is the dimension of the manifold.
    # Note that we use first() to extract the object we care about from the
    # Zygote output.
    J̲̲ = first(Zygote.jacobian(manifold, val))

    # Compute Hessian tensor, i.e., the tensor with the manifold second
    # derivatives. This should be a D×d×d third-order tensor, where D is the
    # dimension of the output space, and d is the dimension of the manifold.
    # Note that we have to manually evaluate the hessian on each dimension of
    # the output.
    H̲̲ = permutedims(
        cat(
            [Zygote.hessian(v -> manifold(v)[D], val) for D = 1:out_dim]...,
            dims=3
        ),
        (3, 1, 2)
    )

    # Compute the derivative of the Riemmanian metric. This should be a d×d×d
    # third-order tensor.
    return @tensor ∂M̲̲[i, j, k] := H̲̲[l, i, k] * J̲̲[l, j] +
                                    H̲̲[l, k, j] * J̲̲[l, i]
end # function

@doc raw"""
    christoffel_simbols(manifold, val, out_dim)

Function to compute the Christoffel symbols from the Riemmanian metric `M̲̲` as
a function of the coordinates in the input space using `Zygote.jl`.

# Arguments
- `manifold::Function`: Function defining the manifold.
- `val::Vector{<:AbstractFloat}`: Value where to evaluate the metric.
- `out_dim::Int`: Dimensionality of the output space. This is required because
  `Zygote.jl` cannot compute the required derivatives automatically, but we have
  to "manually" iterate over the variables in the output space, one at the time.

# Returns
- `Γᵏᵢⱼ::Array{<:AbstractFloat}`: Rank-3 tensor evaluating the Christoffel
  symbols given a Riemmanian manifold metric.
"""
function christoffel_symbols(
    manifold::Function, val::Vector{T}, out_dim::Int
)::Array{T} where {T<:AbstractFloat}
    # Evaluate metric inverse
    M̲̲⁻¹ = LinearAlgebra.inv(riemmanian_metric(manifold, val))

    # Evaluate metric derivative
    ∂M̲̲ = ∂M̲̲∂γ̲(manifold, val, out_dim)

    # Compute Christoffel Symbols
    return @tensor Γᵏᵢⱼ[i, j, k] := (1 / 2) * M̲̲⁻¹[k, h] *
                                    (∂M̲̲[i, h, j] + ∂M̲̲[j, h, i] - ∂M̲̲[i, j, h])
end # function

@doc raw"""
    geodesic_system!(du, u, param, t)

Funtion thefining the right-hand side of the system of geodesic ODEs. The
function is evaluated in place⸺making `du` the first input⸺to accelerate the
integration. To define the 2nd order system of differential equtions we define a
system of coupled 1st order ODEs.

# Arguments
- `du::Array{<:AbstractFloat}`: derivatives of state variables. The first `end ÷
  2` entries define the velocity in latent space where the curve γ is being
  evaluated, i.e., dγ/dt. The second half defines the acceleration of the curve,
  i.e., d²γ/dt².
- `u::Array{<:AbstractFloat}`: State variables. The first `end ÷ 2` entries
  define the coordinates in latent space where the curve γ is being evaluated,
  i.e., γ. The second half defines the velocity of the curve, i.e., dγ/dt.
- `param::Dictionary`: Parameters required for the geodesic differential
  equation system and the boundary value integration. The list of required
  parameters are:
  - `in_dim::Int`: Dimensionality of input space.
  - `out_dim::Int`: Dimensionality of output space.
  - `manifold::Function`: Function definining the manifold.
  - `γ_init::Vector{<:AbstractFloat}`: Initial position in latent space.
  - `γ_end::Vector{<:AbstractFloat}`: Final position in latent space.
- `t::AbstractFloat`: Time where to evaluate the righ-hand side.
"""
function geodesic_system!(du, u, param, t)
    # Extract dimensions d and D
    in_dim, out_dim = param[:in_dim], param[:out_dim]
    # Extract manifold function
    manifold = param[:manifold]

    # Set curve coordinates 
    γ = u[1:in_dim]
    # Set curve velocities
    dγ = u[in_dim+1:end]

    # Compute Christoffel symbols
    Γᵏᵢⱼ = christoffel_symbols(manifold, γ, out_dim)

    # Define the geodesic system of 2nd order ODEs
    @tensor d²γ[k] := -Γᵏᵢⱼ[i, j, k] * dγ[i] * dγ[j]

    # Update derivative values
    du .= [dγ; d²γ]
end # function

@doc raw"""
    boundary_condition!(residual, u, param, t)

Function that evaluates the residuals between the current position and the
desired boundary conditions.

# Arguments
- `residual::Vector{<:AbstractFloat}`: Array containing the residuals between
  the desired boundary conditions and the current state.
- `param::Dictionary`: Parameters required for the geodesic differential
  equation system and the boundary value integration. The list of required
  parameters are:
  - `in_dim::Int`: Dimensionality of input space.
  - `out_dim::Int`: Dimensionality of output space.
  - `manifold::Function`: Function definining the manifold.
  - `γ_init::Vector{<:AbstractFloat}`: Initial position in latent space.
  - `γ_end::Vector{<:AbstractFloat}`: Final position in latent space.
- `t::AbstractFloat`: Time where to evaluate the righ-hand side.
"""
function boundary_condition!(residual, u, param, t)
    # Extract parameters
    in_dim = param[:in_dim]

    # Compute residual for initial position
    @. residual[1:in_dim] = u[1][1:in_dim] - param[:γ_init]

    # Compute residual for final position
    @. residual[in_dim+1:end] = u[end][1:in_dim] - param[:γ_end]
end # function

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Approximating geodesics via Splines
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    min_energy_spline(γ_init, γ_end, manifold, n_points; step_size, stop, tol)

Function that produces an energy-minimizing spline between two points `γ_init`
and `γ_end` on a Riemmanian manifold using the algorithm by Hoffer & Pottmann.

# Arguments
- `γ_init::AbstractVector{T}`: Initial position of curve `γ` on the parameter
  space.
- `γ_end::AbstractVector{T}`: Final position of curve `γ` on the parameter
  space.
- `manifold::Function`: Function defining the Riemmanian manifold on which the
  curve lies.
- `n_points::Int`: Number of points to use for the interpolation.

## Optional arguments
- `step_size::AbstractFloat=1E-3`: Step size used to update curve parameters.
- `stop::Boolean=true`: Boolean indicating if iterations should stop if the
  current point is within certain tolerance value `tol` from the desired final
  point.
- `tol::AbstractFloat=1E-6`: Tolerated difference between desired and current
  final point for an early stop criteria.

# Returns
- `γ::Matrix{T}`: `d × N` matrix where each row represents each of the
  dimensions on the manifold and `N` is the number of points needed to
  interpolate between points.

where `T <: AbstractFloat`
"""
function min_energy_spline(
    γ_init::AbstractVector{T},
    γ_end::AbstractVector{T},
    manifold::Function,
    n_points::Int;
    step_size::AbstractFloat=1E-3,
    stop::Bool=true,
    tol::AbstractFloat=1E-3
) where {T<:AbstractFloat}
    # Initialize matrix where to store interpolation points
    γ = Matrix{T}(undef, length(γ_init), n_points)
    # Set initial value
    γ[:, 1] .= γ_init

    # Map final position to output space
    x_end = manifold(γ_end)

    # Initialize loop counter
    count = 1
    # Loop through points
    for i = 2:n_points
        # 1. Map current point to output space
        x_current = manifold(γ[:, i-1])
        # 2. Compute Jacobian. Each column represents one basis vector c̲ for
        #    the tangent space at x_current.
        J̲̲ = first(Zygote.jacobian(manifold, γ[:, i-1]))
        # 3. Compute the Riemmanian metric tensor at current point
        M̲̲ = J̲̲' * J̲̲
        # 4. Compute the difference between the current position and the desired
        #    final point
        Δx̲ = x_end .- x_current
        # 5. Compute r̲ vector where each entry is the inner product beetween
        #    Δx̲ and each of the basis vectors of the tangent space c̲. Since
        #    these vectors are stored in the Jacobian, we can simply do matrix
        #    vector multiplication.
        r̲ = J̲̲' * Δx̲
        # 6. Solve linear system to find tangential vector
        t̲ = M̲̲ \ r̲
        # 7. Update parameters given chosen step size
        γ[:, i] = γ[:, i-1] .+ (step_size .* t̲)

        # Update loop counter
        count += 1

        # Check if final point satisfies tolerance
        if (stop) .& (LinearAlgebra.norm(γ[:, i] - γ_end) ≤ tol)
            break
        end # if
    end # for

    # Return curve
    if stop
        return γ[:, 1:count]
    else
        return γ
    end # if
end # function


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Discretized curv characteristics
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    curve_energy(γ̲, manifold)

Function to compute the (discretized) integral defining the energy of a curve γ
on a Riemmanina manifold. The energy is defined as

    E = 1 / 2 ∫ dt ⟨γ̲̇(t), M̲̲ γ̲̇(t)⟩,

where γ̲̇(t) defines the velocity of the parametric curve, and M̲̲ is the
Riemmanian metric. For this function, we use finite differences from the curve
sample points `γ` to compute this integral.

# Arguments
- `γ::AbstractMatrix{T}`: `d×N` long vector where `d` is the dimension of the
  manifold on which the curve lies and `N` is the number of points along the
  curve (without the initial point `γ̲ₒ`). This vector represents the sampled
  poinst along the curve. The longer the number of entries in this vector, the
  more accurate the energy estimate will be. NOTE: Notice that this function
  asks 
- `manifold::Function`: Function defining the Riemmanian manifold on which the
  curve lies.

# Returns
- `Energy::T`: Value of the energy for the path on the manifold.
"""
function curve_energy(γ::AbstractMatrix{T}, manifold::Function) where {T<:Real}
    # Define Δt
    Δt = 1 / size(γ, 2)

    # Compute the differences between points
    Δγ = diff(γ, dims=2)

    # Evaluate and return energy
    return (1 / (2 * Δt)) * sum(
        [
        LinearAlgebra.dot(
            Δγ[:, i], riemmanian_metric(manifold, γ[:, i+1]), Δγ[:, i]
        )
        for i = 1:size(Δγ, 2)
    ]
    )
end # function

@doc raw"""
    curve_length(γ̲, manifold)

Function to compute the (discretized) integral defining the length of a curve γ
on a Riemmanina manifold. The energy is defined as

    E = ∫ dt √(⟨γ̲̇(t), M̲̲ γ̲̇(t)⟩),

where γ̲̇(t) defines the velocity of the parametric curve, and M̲̲ is the
Riemmanian metric. For this function, we use finite differences from the curve
sample points `γ` to compute this integral.

# Arguments
- `γ::AbstractMatrix{T}`: `d×N` long vector where `d` is the dimension of the
  manifold on which the curve lies and `N` is the number of points along the
  curve (without the initial point `γ̲ₒ`). This vector represents the sampled
  poinst along the curve. The longer the number of entries in this vector, the
  more accurate the energy estimate will be. NOTE: Notice that this function
  asks 
- `manifold::Function`: Function defining the Riemmanian manifold on which the
  curve lies.

# Returns
- `Length::T`: Value of the Length for the path on the manifold.
"""
function curve_length(γ::AbstractMatrix{T}, manifold::Function) where {T<:Real}
    # Define Δt
    Δt = 1 / size(γ, 2)

    # Compute the differences between points
    Δγ = diff(γ, dims=2)

    # Evaluate and return energy
    return sum(
        [
        sqrt(
            LinearAlgebra.dot(
                Δγ[:, i], riemmanian_metric(manifold, γ[:, i+1]), Δγ[:, i]
            )
        )
        for i = 1:size(Δγ, 2)
    ]
    )
end # function

@doc raw"""
    curve_lengths(γ̲, manifold)

Function to compute the (discretized) integral defining the length of a curve γ
on a Riemmanina manifold. The energy is defined as

    E = ∫ dt √(⟨γ̲̇(t), M̲̲ γ̲̇(t)⟩),

where γ̲̇(t) defines the velocity of the parametric curve, and M̲̲ is the
Riemmanian metric. For this function, we use finite differences from the curve
sample points `γ` to compute this integral.

# Arguments
- `γ::AbstractMatrix{T}`: `d×N` long vector where `d` is the dimension of the
  manifold on which the curve lies and `N` is the number of points along the
  curve (without the initial point `γ̲ₒ`). This vector represents the sampled
  poinst along the curve. The longer the number of entries in this vector, the
  more accurate the energy estimate will be. NOTE: Notice that this function
  asks 
- `manifold::Function`: Function defining the Riemmanian manifold on which the
  curve lies.

# Returns
- `Lengths::Vector{T}`: Value of the lengths for each individual path on the
  manifold.
"""
function curve_lengths(γ::AbstractMatrix{T}, manifold::Function) where {T<:Real}
    # Define Δt
    Δt = 1 / size(γ, 2)

    # Compute the differences between points
    Δγ = diff(γ, dims=2)

    # Evaluate and return energy
    return [
        sqrt(
            LinearAlgebra.dot(
                Δγ[:, i], riemmanian_metric(manifold, γ[:, i+1]), Δγ[:, i]
            )
        )
        for i = 1:size(Δγ, 2)
    ]
end # function