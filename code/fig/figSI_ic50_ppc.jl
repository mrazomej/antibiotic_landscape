## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic

# Import package to handle DataFrames
import DataFrames as DF
import CSV

# Import Glob to list files
import Glob

# Load CairoMakie for plotting
using CairoMakie
import ColorSchemes

# Import basic math libraries
import StatsBase
import LinearAlgebra
import Random

# Activate backend
CairoMakie.activate!()

# Set PBoC Plotting style
Antibiotic.viz.theme_makie!()

## =============================================================================

# Define output directory
fig_dir = "$(git_root())/fig/supplementary"

# Generate output directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating output directory...")
    mkpath(fig_dir)
end

## =============================================================================

println("Loading raw data...")

# Load data into a DataFrame
df = CSV.read(
    "$(git_root())/data/Iwasawa_2022/iwasawa_tidy.csv", DF.DataFrame
)

# Remove blank measurements
df = df[.!df.blank, :]
# Remove zero concentrations
df = df[df.concentration_ugmL.>0, :]

## =============================================================================

println("Define logistic function fit to data...")

@doc raw"""
    logistic(x, a, b, c, ic50)

Compute the logistic function used to model the relationship between antibiotic
concentration and bacterial growth.

This function implements the following equation:

f(x) = a / (1 + (x / ic50)^b) + c

# Arguments
- `x`: Antibiotic concentration (input variable)
- `a`: Maximum effect parameter (difference between upper and lower asymptotes)
- `b`: Slope parameter (steepness of the curve)
- `c`: Minimum effect parameter (lower asymptote)
- `ic50`: IC₅₀ parameter (concentration at which the effect is halfway between
  the minimum and maximum)

# Returns
The computed effect (e.g., optical density) for the given antibiotic
concentration and parameters.

Note: This function is vectorized and can handle array inputs for `x`.
"""
function logistic(logx, a, b, c, logic50)
    return @. a / (1.0 + exp(b * (logx - logic50))) + c
end

## =============================================================================

println("Define function to compute quantile ranges...")

@doc raw"""
matrix_quantile_range(quantile, matrix; dim=2) 

Function to compute the quantile ranges of matrix `matrix` over dimension `dim`.

For example, if `quantile[1] = 0.95`, this function returns the `0.025` and 
`0.975` quantiles that capture 95 percent of the entries in the matrix.

# Arguments
- `quantile::Vector{<:AbstractFloat}`: List of quantiles to extract from the
  posterior predictive checks.  
- `matrix::Matrix{<:Real}`: Array over which to compute quantile ranges.

# Keyword Arguments
- `dim::Int=2`: Dimension over which to compute quantiles. Default is 2, i.e. 
  columns.

# Returns
- `qs`: Matrix with requested quantiles over specified dimension.
"""
function matrix_quantile_range(
    quantile::Vector{<:AbstractFloat}, matrix::Matrix{T}; dims::Int=2
) where {T<:Real}
    # Check that all quantiles are within bounds
    if any(.![0.0 ≤ x ≤ 1 for x in quantile])
        error("All quantiles must be between zero and one")
    end # if

    # Check that dim corresponds to a matrix
    if (dims != 1) & (dims != 2)
        error("Dimensions should match a Matrix dimensiosn, i.e., 1 or 2")
    end # if

    # Get opposite dimension   
    op_dims = first([1, 2][[1, 2].∈Set(dims)])

    # Initialize array to save quantiles
    array_quantile = Array{T}(undef, size(matrix, op_dims), length(quantile), 2)

    # Loop through quantile
    for (i, q) in enumerate(quantile)
        # Lower bound
        array_quantile[:, i, 1] = StatsBase.quantile.(
            eachslice(matrix, dims=dims), (1.0 - q) / 2.0
        )
        # Upper bound
        array_quantile[:, i, 2] = StatsBase.quantile.(
            eachslice(matrix, dims=dims), 1.0 - (1.0 - q) / 2.0
        )
    end # for

    return array_quantile

end # function


## =============================================================================

println("List MCMC samples...")

# Define output directory
out_dir = "$(git_root())/data/Iwasawa_2022/mcmc_nonnegative"

# List all files in the output directory
files = sort(Glob.glob("$(out_dir)/*.csv"[2:end], "/"))

# Initialize empty dataframe to store metadata
df_meta = DF.DataFrame()

# Loop over each file and extract metadata
for file in files
    # Extract antibiotic from filename using regular expressions
    antibiotic = match(r"_(\w+)antibiotic", file).captures[1]

    # Extract day from filename using regular expressions
    day = parse(Int, match(r"_(\d+)day", file).captures[1])

    # Extract strain from filename using regular expressions
    strain_num = parse(Int, match(r"_(\d+)strain", file).captures[1])

    # Extract design from filename using regular expressions
    design = parse(Int, match(r"_(\d+)design", file).captures[1])

    # Extract environment from filename using regular expressions
    env = match(r"design_(\w+)env", file).captures[1]

    # Create a new row with the extracted metadata
    DF.append!(df_meta, DF.DataFrame(
        antibiotic=antibiotic,
        day=day,
        strain_num=strain_num,
        design=design,
        env=env,
        file=file
    ))
end

println("Number of MCMC samples: $(size(df_meta, 1))")

## =============================================================================


println("Plot example fits...")

Random.seed!(42)

# Define number of rows and columns
rows = 4
cols = 4

# Define quantiles to compute
quantiles = [0.95, 0.68, 0.5]

# Initialize figure
fig = Figure(size=(200 * cols, 200 * rows))

# ------------------------------------------------------------------------------

# Add global grid layout
gl = fig[1, 1] = GridLayout()

# Add grid layout for banner
gl_banner = gl[1, 1] = GridLayout()

# Add grid layout for plots
gl_plot = gl[2, 1] = GridLayout()

# ------------------------------------------------------------------------------

# Add box for section title
Box(
    gl_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-75, right=-40) # Moves box to the left and right
)

# Add section title
Label(
    gl_banner[1, 1],
    "posterior predictive checks for logistic model",
    fontsize=14,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
    alignmode=Mixed(; left=-40) # Moves text to the left
)

# ------------------------------------------------------------------------------

# Select random indexes
idxs = Random.randperm(size(df_meta, 1))[1:rows*cols]

# Loop through each plot
for i in 1:rows*cols
    # Locate row and column index
    row_idx = (i - 1) ÷ cols + 1
    col_idx = (i - 1) % cols + 1

    # Extract row to analyze
    row = df_meta[idxs[i], :]

    # Add axis
    ax = Axis(
        gl_plot[row_idx, col_idx],
        xlabel="[antibiotic] (μg/mL)",
        ylabel="optical density",
        title="$(replace(row.env, "_" => " ")) | day $(row.day)",
        xscale=log10,
        yscale=log10,
        xlabelsize=12,
        ylabelsize=12,
        xticklabelsize=12,
        yticklabelsize=12,
        titlesize=12,
    )

    # Load the file
    chain = CSV.read(row.file, DF.DataFrame)

    # Extract corresponding data from the raw data
    data = df[
        (df.antibiotic.==row.antibiotic).&(df.day.==row.day).&(df.env.==row.env).&(df.strain_num.==row.strain_num),
        :]
    # Sort data by concentration
    sort!(data, :concentration_ugmL)

    # Locate unique concentrations
    unique_concentrations = unique(data.concentration_ugmL)

    # Initialize matrix to store samples
    y_samples = Array{Float64}(
        undef, length(unique_concentrations), size(chain, 1)
    )

    # Loop through samples
    for i in 1:size(chain, 1)
        logy_samples = log.(logistic(
            log.(unique_concentrations),
            chain[i, :a],
            chain[i, :b],
            chain[i, :c],
            chain[i, :logic50]
        ))
        # Add noise
        logy_samples .+= randn(length(unique_concentrations)) * √(chain[i, :σ²])
        # Store samples
        y_samples[:, i] = exp.(logy_samples)
    end # for

    # Compute quantiles
    y_quantiles = matrix_quantile_range(quantiles, y_samples, dims=1)

    # Define colors for quantiles
    colors = get(ColorSchemes.Blues_9, range(0.3, 0.6, length(quantiles)))

    # Loop through quantiles
    for i in 1:length(quantiles)
        # Plot quantile
        band!(
            ax,
            unique_concentrations,
            y_quantiles[:, i, 1],
            y_quantiles[:, i, 2],
            color=colors[i]
        )
    end # for

    # Plot data
    scatter!(
        ax,
        data.concentration_ugmL,
        data.OD,
        color=:black
    )

end # for

# Save figure
save("$(fig_dir)/figSI_ic50_ppc.png", fig)
save("$(fig_dir)/figSI_ic50_ppc.pdf", fig)

fig