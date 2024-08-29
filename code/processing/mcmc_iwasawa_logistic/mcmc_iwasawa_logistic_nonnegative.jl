## -----------------------------------------------------------------------------
println("Import packages")

# Import packages

# Import project package
import Antibiotic

# Import package to handle DataFrames
import DataFrames as DF
import CSV

# Import library for Bayesian inference
import Turing

# Import library to list files
import Glob
import IterTools

# Import packages to work with data
import DataFrames as DF

# Import basic math libraries
import StatsBase
import LinearAlgebra
import Random
import LsqFit

## -----------------------------------------------------------------------------

println("Set random seed")
Random.seed!(42)

println("Define output directory")
out_dir = "$(git_root())/data/Iwasawa_2022/mcmc_nonnegative"

# Check if directory exists
if !isdir(out_dir)
    # Create directory
    mkdir(out_dir)
end

## -----------------------------------------------------------------------------

println("Load data")

# Load data into a DataFrame
df = CSV.read(
    "$(git_root())/data/Iwasawa_2022/iwasawa_tidy.csv", DF.DataFrame
)

# Remove blank measurements
df = df[.!df.blank, :]
# Remove zero concentrations
df = df[df.concentration_ugmL.>0, :]

## -----------------------------------------------------------------------------

println("Group data")
# Group data by :antibiotic, :env, :day
df_group = DF.groupby(df, [:antibiotic, :env, :day])

## -----------------------------------------------------------------------------

println("Define logistic model")

@doc raw"""
    logistic(x, a, b, c, ic50)

Compute the logistic function used to model the relationship between antibiotic
concentration and bacterial growth.

This function implements the following equation:

f(x) = a / (1 + exp(b * (logx - logic50))) + c

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

# Define second method that takes vector of parameters
function logistic(logx, params)
    return logistic(logx, params...)
end


## -----------------------------------------------------------------------------

println("Define outlier detection function")

# Function to fit logistic model and detect outliers
function fit_logistic_and_detect_outliers(logx, y; threshold=2)
    # Initial parameter guess
    p0 = [0.1, 1.0, maximum(y) - minimum(y), StatsBase.median(logx)]

    # Fit the logistic model
    fit = LsqFit.curve_fit(logistic, logx, y, p0)

    # Calculate residuals
    residuals = y - logistic(logx, fit.param)

    # Calculate standard deviation of residuals
    σ = std(residuals)

    # Identify outliers
    outliers_idx = abs.(residuals) .> threshold * σ

    # Return outlier indices
    return outliers_idx
end

## -----------------------------------------------------------------------------

println("Define Turing model")

Turing.@model function logistic_model(
    logx, logy, prior_params::NamedTuple=NamedTuple()
)
    # Define default prior parameters
    default_params = (
        logic50=(0, 1),
        a=(0, 1),
        b=(0, 1),
        c=(0, 1),
        σ²=(0, 1),
    )

    # Merge default parameters with provided parameters
    params = merge(default_params, prior_params)

    # Define priors
    logic50 ~ Turing.Normal(params.logic50...)
    a ~ Turing.LogNormal(params.a...)
    b ~ Turing.LogNormal(params.b...)
    c ~ Turing.truncated(Turing.Normal(params.c...), 0, Inf)
    σ² ~ Turing.truncated(Turing.Normal(params.σ²...), 0, Inf)

    # Compute model predictions
    μ = log.(logistic(logx, a, b, c, logic50))

    # Define likelihood
    logy ~ Turing.MvNormal(
        μ,
        LinearAlgebra.Diagonal(fill(σ², length(logx)))
    )
end

## -----------------------------------------------------------------------------

println("Performing MCMC on data")

# Define prior parameters
prior_params = (
    a=(log(0.1), 0.1),
    b=(0, 1),
    c=(0, 1),
    σ²=(0, 0.1),
)

# Define number of steps
n_burnin = 10_000
n_samples = 1_000

# Loop through each group
for (i, data) in enumerate(df_group[end:-1:1])
    println("Performing MCMC on group $i / $(length(df_group))")
    # Define ouput file name
    out_file = "$out_dir/chain_" *
               "$(first(data.antibiotic))antibiotic_" *
               "$(first(data.day))day_" *
               "$(first(data.strain_num))strain_" *
               "$(first(data.design))design_" *
               "$(first(data.env))env.csv"
    # Check if file exists
    if isfile(out_file)
        println("Skipping MCMC because file exists...")
        continue
    end

    # Clean data
    logx = log.(data.concentration_ugmL)
    y = data.OD
    outliers_idx = fit_logistic_and_detect_outliers(logx, y)
    data_clean = data[.!outliers_idx, :]

    # Define model
    model = logistic_model(
        log.(data_clean.concentration_ugmL),
        log.(data_clean.OD),
        prior_params,
    )

    # Perform MCMC
    chain = Turing.sample(
        model,
        Turing.NUTS(),
        Turing.MCMCThreads(),
        n_burnin + n_samples,
        4,
        progress=true
    )
    # Remove burnin
    chain = chain[n_burnin+1:end, :, :]

    # Save chain
    CSV.write(out_file, DF.DataFrame(chain))
end