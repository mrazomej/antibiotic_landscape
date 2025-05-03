import AutoEncoderToolkit as AET
import Flux

"""
    trajectory_velocity_finitediff(trajectory::AbstractMatrix)

Compute the velocity of a trajectory using finite differences.

This function calculates the velocity between consecutive points in a trajectory
using a forward finite difference approximation.

# Arguments
- `trajectory::AbstractMatrix`: Matrix representing the trajectory points. Each
  column represents a point along the trajectory and each row represents a
  dimension.

# Returns
- `velocity::Matrix`: Matrix of velocity vectors. The shape is the same as the
  input trajectory except there is one fewer point.

# Details
The velocity is computed as the difference between consecutive points in the
    trajectory: v_i = (x_{i+1} - x_i)
"""
function trajectory_velocity_finitediff(trajectory::AbstractMatrix)
    # Calculate velocities using forward differences
    velocity = trajectory[:, 2:end] - trajectory[:, 1:end-1]

    return velocity
end

# ------------------------------------------------------------------------------

"""
    trajectory_length(
        trajectory::AbstractMatrix, 
        metric_tensor_fn::Function
    )

Calculate the length of a trajectory on a Riemannian manifold.

This function approximates the length of a curve on a Riemannian manifold using a
discretized version of the curve length integral:

    L(γ) = ∫ √(γ̇(t)ᵀ G(γ(t)) γ̇(t)) dt

where γ(t) is the curve, γ̇(t) is its velocity, and G(γ(t)) is the Riemannian 
metric tensor at point γ(t).

# Arguments
- `trajectory::AbstractMatrix`: Matrix representing the trajectory points. Each
  column represents a point along the trajectory and each row represents a dimension.
- `metric_tensor_fn::Function`: Function that takes a point and returns the 
  Riemannian metric tensor at that point. The signature should be:
  `metric_tensor_fn(point) -> Matrix`.

# Returns
- `length::Float64`: The approximate length of the trajectory on the Riemannian manifold.

# Notes
This function uses forward finite differences to approximate the velocity at each point.
The integral is approximated using a simple Riemann sum.
"""
function trajectory_length(
    trajectory::AbstractMatrix,
    metric_tensor_fn::Function
)
    # Compute velocity using finite differences
    velocity = trajectory_velocity_finitediff(trajectory)

    # Number of segments (one less than number of points)
    n_segments = size(trajectory, 2) - 1

    # Initialize length
    length = 0.0

    # For each segment, compute the contribution to the length
    for i in 1:n_segments
        # Get midpoint of segment (for evaluating metric tensor)
        point_i = trajectory[:, i]
        point_i_plus_1 = trajectory[:, i+1]
        midpoint = (point_i + point_i_plus_1) / 2
        v_i = velocity[:, i]

        # Get metric tensor at midpoint
        G_i = metric_tensor_fn(midpoint)

        # Compute segment length: √(v_i^T G_i v_i)
        segment_length = sqrt(LinearAlgebra.dot(v_i, G_i * v_i))

        # Add to total length
        length += segment_length
    end

    return length
end

# ------------------------------------------------------------------------------

"""
    trajectory_length_rhvae(
        trajectory::AbstractMatrix, 
        rhvae
    )

Calculate the length of a trajectory in the RHVAE latent space.

This function is a convenience wrapper around `trajectory_length` that uses the
metric tensor from an RHVAE model.

# Arguments
- `trajectory::AbstractMatrix`: Matrix representing the trajectory points. Each column 
  represents a point along the trajectory and each row represents a dimension.
- `rhvae`: The RHVAE model containing the Riemannian metric.

# Returns
- `length::Float64`: The approximate length of the trajectory on the RHVAE latent space.

# Notes
This function evaluates the RHVAE metric tensor at each midpoint of consecutive trajectory
points and computes the length accordingly.
"""
function trajectory_length_rhvae(
    trajectory::AbstractMatrix,
    rhvae
)
    # Define function to compute metric tensor using RHVAE
    function metric_tensor_fn(point)
        # Reshape point to column vector for RHVAE 
        point_col = reshape(point, :, 1)

        # Get metric tensor using RHVAE
        return AET.RHVAEs.G_inv(point_col, rhvae)
    end

    # Compute trajectory length
    return trajectory_length(trajectory, metric_tensor_fn)
end

# ------------------------------------------------------------------------------

"""
    vec_mat_vec_batched(v1::AbstractMatrix, M::AbstractArray, v2::AbstractMatrix)

Compute the batched vector-matrix-vector product v1ᵀ M v2.

This is a utility function for computing v1ᵀ M v2 for batches of vectors and
matrices. Similar to the one in your existing code.

# Arguments
- `v1::AbstractMatrix`: First batch of vectors. Shape (dim, batch_size)
- `M::AbstractArray`: Batch of matrices. Shape (dim, dim, batch_size)
- `v2::AbstractMatrix`: Second batch of vectors. Shape (dim, batch_size)

# Returns
- `result::Vector`: The result of v1ᵀ M v2 for each element in the batch. Shape
  (batch_size,)
"""
function vec_mat_vec_batched(
    v::AbstractMatrix,
    M::AbstractArray,
    w::AbstractMatrix
)
    # Compute v̲ M̲̲ w̲ in a broadcasted manner
    return vec(sum(v .* Flux.batched_vec(M, w), dims=1))
end # function

# ------------------------------------------------------------------------------

"""
    curve_energy(
        trajectory::AbstractMatrix, 
        metric_tensor_fn::Function
    )

Calculate the energy of a trajectory on a Riemannian manifold.

The energy of a curve γ on a Riemannian manifold is defined as:

    E(γ) = ∫ γ̇(t)ᵀ G(γ(t)) γ̇(t) dt

This can be thought of as the sum of the squared infinitesimal lengths along the
curve.

# Arguments
- `trajectory::AbstractMatrix`: Matrix representing the trajectory points. Each
  column represents a point along the trajectory and each row represents a
  dimension.
- `metric_tensor_fn::Function`: Function that takes a point and returns the
  Riemannian metric tensor at that point.

# Returns
- `energy::Float64`: The energy of the trajectory on the Riemannian manifold.
"""
function curve_energy(
    trajectory::AbstractMatrix,
    metric_tensor_fn::Function
)
    # Compute velocity using finite differences
    velocity = trajectory_velocity_finitediff(trajectory)

    # Number of segments
    n_segments = size(trajectory, 2) - 1

    # Initialize energy
    energy = 0.0

    # For each segment, compute the contribution to the energy
    for i in 1:n_segments
        # Get midpoint of segment (for evaluating metric tensor)
        point_i = trajectory[:, i]
        point_i_plus_1 = trajectory[:, i+1]
        midpoint = (point_i + point_i_plus_1) / 2
        v_i = velocity[:, i]

        # Get metric tensor at midpoint
        G_i = metric_tensor_fn(midpoint)

        # Compute segment energy: v_i^T G_i v_i
        segment_energy = LinearAlgebra.dot(v_i, G_i * v_i)

        # Add to total energy
        energy += segment_energy
    end

    return energy / 2
end