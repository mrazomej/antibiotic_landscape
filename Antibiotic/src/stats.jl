import StatsBase
import LinearAlgebra


@doc raw"""
    procrustes(X, Y; dims=2, center=true)

Perform Procrustes analysis to find the optimal rigid transformation between two
sets of points.

Procrustes analysis finds a linear transformation (translation, rotation, and
uniform scaling) of the points in X to best conform them to the points in Y,
minimizing the sum of squared differences.

# Arguments
- `X::AbstractMatrix`: First set of points
- `Y::AbstractMatrix`: Second set of points
- `dims::Int=2`: Dimension along which points are stored. If dims=2 (default),
  each column represents a point. If dims=1, each row represents a point.
- `center::Bool=true`: Whether to center the data by subtracting means

# Returns
- `X_transformed::Matrix`: The transformed X matrix aligned with Y
- `R::Matrix`: The rotation matrix used to transform X
- `s::Float64`: The scaling factor used to transform X
- `correlation::Float64`: The Procrustes correlation (goodness-of-fit) between 0
  and 1

# Details
The analysis follows these steps:
1. Center both point sets by subtracting their means (if center=true)
2. Compute the optimal rotation matrix using SVD
3. Apply rotation and scaling to align X with Y
4. Calculate a correlation-like goodness-of-fit statistic

The correlation ranges from 0 (no similarity) to 1 (perfect similarity).
"""
function procrustes(X, Y; dims=2, center=true)
    if center
        # Center the data
        X_centered = X .- StatsBase.mean(X, dims=dims)
        Y_centered = Y .- StatsBase.mean(Y, dims=dims)
    else
        X_centered = X
        Y_centered = Y
    end

    if dims == 1
        # Compute the inner product matrix
        A = LinearAlgebra.transpose(Y_centered) * X_centered
    elseif dims == 2
        # Compute the inner product matrix
        A = Y_centered * LinearAlgebra.transpose(X_centered)
    else
        throw(ArgumentError("dims must be 1 or 2"))
    end

    # Singular value decomposition
    U, S, Vt = LinearAlgebra.svd(A)

    # Compute the rotation matrix
    R = U * Vt

    # Compute the scaling factor
    norm_X = sqrt(sum(X_centered .^ 2))
    norm_Y = sqrt(sum(Y_centered .^ 2))
    s = sum(S) / (norm_X^2)  # This is the optimal scaling factor

    if dims == 1
        # Transform X to align with Y (with scaling)
        X_transformed = s * (X_centered * R)
    elseif dims == 2
        # Transform X to align with Y (with scaling)
        X_transformed = s * (R * X_centered)
    else
        throw(ArgumentError("dims must be 1 or 2"))
    end

    # Calculate the Procrustes statistic (correlation-like measure)
    # Calculate the scaled Procrustes statistic (between 0 and 1)
    trace_S = sum(S)
    correlation = trace_S / (sqrt(norm_X * norm_Y))

    return X_transformed, R, s, correlation
end

# ------------------------------------------------------------------------------

"""
    discrete_frechet_distance(P, Q; dims=2)

Calculate the discrete Fréchet distance between two trajectories P and Q.

The Fréchet distance is a measure of similarity between curves that takes into
account the location and ordering of the points along the curves. It is often
described as the minimum length of a leash required to connect a dog and its
owner as they walk along their respective curves, being allowed to vary their
speeds but not go backwards.

# Arguments
- `P::Matrix`: First trajectory matrix
- `Q::Matrix`: Second trajectory matrix
- `dims::Int=2`: Dimension along which points are stored. If dims=2 (default),
  each column represents a point. If dims=1, each row represents a point.

# Returns
- `Float64`: The discrete Fréchet distance between trajectories P and Q

# Details
The function implements the dynamic programming approach to compute the discrete
Fréchet distance. It builds a coupling measure matrix that stores the Fréchet
distances between all pairs of prefixes of the input trajectories. The final
value in the matrix represents the Fréchet distance between the complete
trajectories.

The algorithm has a time complexity of O(nm) where n and m are the lengths of
the input trajectories P and Q respectively.
"""
function discrete_frechet_distance(P, Q; dims=2)
    if dims == 1
        # If dims=1, points are rows, so use number of rows
        n = size(P, 1)
        m = size(Q, 1)
    elseif dims == 2
        # If dims=2, points are columns, so use number of columns
        n = size(P, 2)
        m = size(Q, 2)
    else
        throw(ArgumentError("dims must be 1 or 2"))
    end

    # Initialize the coupling measure matrix
    cm = fill(Inf, n, m)

    # Compute the distance between the first points
    if dims == 1
        cm[1, 1] = LinearAlgebra.norm(P[1, :] - Q[1, :])
    else
        cm[1, 1] = LinearAlgebra.norm(P[:, 1] - Q[:, 1])
    end

    # Fill the first column
    for i in 2:n
        if dims == 1
            cm[i, 1] = max(cm[i-1, 1], LinearAlgebra.norm(P[i, :] - Q[1, :]))
        else
            cm[i, 1] = max(cm[i-1, 1], LinearAlgebra.norm(P[:, i] - Q[:, 1]))
        end
    end

    # Fill the first row
    for j in 2:m
        if dims == 1
            cm[1, j] = max(cm[1, j-1], LinearAlgebra.norm(P[1, :] - Q[j, :]))
        else
            cm[1, j] = max(cm[1, j-1], LinearAlgebra.norm(P[:, 1] - Q[:, j]))
        end
    end

    # Fill the rest of the matrix using dynamic programming
    for i in 2:n
        for j in 2:m
            if dims == 1
                point_distance = LinearAlgebra.norm(P[i, :] - Q[j, :])
            else
                point_distance = LinearAlgebra.norm(P[:, i] - Q[:, j])
            end
            cm[i, j] = max(
                min(cm[i-1, j], cm[i-1, j-1], cm[i, j-1]),
                point_distance
            )
        end
    end

    # Return the Fréchet distance
    return cm[n, m]
end