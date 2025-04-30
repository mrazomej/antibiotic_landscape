import LinearAlgebra
import Flux
import StatsBase

"""
    svd_cross_validation(data_matrix::Matrix{Float64}, 
                         row_split_idx::Int, 
                         col_split_idx::Int, 
                         max_rank::Int=0)

Perform bi-cross validation on a data matrix using SVD decomposition as
described in Owen & Perry (2009).

The method splits the data matrix 𝐗 into four submatrices:

𝐗 = [𝐀 𝐁;
     𝐂 𝐃]

It can be shown that:

𝐀 = 𝐁 𝐃⁺ 𝐂

where 𝐃⁺ is the Moore-Penrose pseudo-inverse of matrix 𝐃.

The function computes rank-𝑟 approximations of 𝐃 to test the accuracy of lower
rank models in predicting the left-out data in 𝐀:

1. Perform SVD on 𝐃: 𝐃 = 𝐔 𝚺 𝐕ᵀ
2. Create rank-𝑟 approximation of 𝐃: 𝐃ᵣ = 𝐔 𝚺ᵣ 𝐕ᵀ 
   where 𝚺ᵣ keeps only the top 𝑟 singular values
3. Compute the predicted matrix: 𝐀̂ = 𝐁 𝐃ᵣ⁺ 𝐂
4. Calculate R² and MSE to evaluate prediction quality

# Arguments
- `data_matrix::Matrix{Float64}`: The data matrix 𝐗 to perform cross-validation
  on
- `row_split_idx::Int`: Index to split the rows (𝐀 and 𝐁 will include rows
  1:row_split_idx)
- `col_split_idx::Int`: Index to split the columns (𝐀 and 𝐂 will include
  columns 1:col_split_idx)
- `max_rank::Int=0`: Maximum rank to test. If 0, set to min(size(𝐃))

# Returns
- `r2_values::Vector{Float64}`: R² values for each rank (1:max_rank)
- `mse_values::Vector{Float64}`: Mean squared error values for each rank (1:max_rank)
- `predictions::Vector{Matrix{Float64}}`: Predicted 𝐀 matrices for each rank

# References
- Owen, A. B., & Perry, P. O. (2009). Bi-cross-validation of the SVD and the
  nonnegative matrix factorization. The Annals of Applied Statistics, 3(2),
  564-594.
"""
function svd_cross_validation(
    data_matrix::Matrix{<:AbstractFloat},
    row_split_idx::Int,
    col_split_idx::Int,
    max_rank::Int=0
)
    # Get size of data matrix
    n_rows, n_cols = size(data_matrix)

    # Ensure split indices are valid
    row_split_idx = min(max(1, row_split_idx), n_rows - 1)
    col_split_idx = min(max(1, col_split_idx), n_cols - 1)

    # Define submatrices
    A_mat = @view data_matrix[1:row_split_idx, 1:col_split_idx]
    B_mat = @view data_matrix[1:row_split_idx, (col_split_idx+1):n_cols]
    C_mat = @view data_matrix[(row_split_idx+1):n_rows, 1:col_split_idx]
    D_mat = @view data_matrix[(row_split_idx+1):n_rows, (col_split_idx+1):n_cols]

    # Convert views to matrices for SVD
    D_matrix = Matrix(D_mat)

    # Determine maximum rank to test
    if max_rank ≤ 0
        max_rank = min(size(D_matrix)...)
    else
        max_rank = min(max_rank, min(size(D_matrix)...))
    end

    # Perform SVD on D matrix
    U_D, Σ_D, V_D = LinearAlgebra.svd(D_matrix)

    # Initialize results
    r2_values = zeros(eltype(data_matrix), max_rank)
    mse_values = zeros(eltype(data_matrix), max_rank)
    predictions = Vector{Matrix{eltype(data_matrix)}}(undef, max_rank)

    # Loop through ranks
    for r in 1:max_rank
        # Create rank-r approximation of D
        Σ_r = zeros(length(Σ_D))
        Σ_r[1:r] = Σ_D[1:r]
        D_r = U_D * LinearAlgebra.Diagonal(Σ_r) * V_D'

        # Predict A using B, C, and pseudo-inverse of D_r
        A_pred = B_mat * LinearAlgebra.pinv(D_r) * C_mat

        # Store prediction
        predictions[r] = A_pred

        # Calculate R²
        error = A_mat - A_pred
        ss_total = sum((A_mat .- StatsBase.mean(A_mat)) .^ 2)
        ss_residual = sum(error .^ 2)
        r2_values[r] = 1 - ss_residual / ss_total

        # Calculate MSE
        mse_values[r] = Flux.mse(A_mat, A_pred)
    end

    return r2_values, mse_values, predictions
end