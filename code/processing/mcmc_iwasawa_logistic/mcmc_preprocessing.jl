## =============================================================================
println("Loading libraries...")

# Import libraries to handel data
import CSV
import DataFrames as DF
import Glob
import JLD2

# Import basic math
import StatsBase
import Random
Random.seed!(42)

## =============================================================================

println("Listing MCMC files...")

# Define output directory
out_dir = "$(git_root())/output/mcmc_iwasawa_logistic"

# List all files in the output directory
files = sort(Glob.glob("$(out_dir)/chain*.csv"[2:end], "/"))

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

println("Computing IC50 credible regions...")

# Initialize dataframe to store results
df_ic50 = DF.DataFrame()

# Loop over each file
for row in eachrow(df_meta)
    # Read file
    chain = CSV.read(row.file, DF.DataFrame)

    # Compute mean and 95% credible interval
    logic50_mean = StatsBase.mean(chain[:, :logic50])
    logic50_ci = StatsBase.quantile(chain[:, :logic50], [0.025, 0.975])

    # Add to dataframe
    DF.append!(df_ic50, DF.DataFrame(
        drug=row.antibiotic,
        day=row.day,
        strain_num=row.strain_num,
        design=row.design,
        env=row.env,
        logic50_mean=logic50_mean,
        logic50_ci_lower=logic50_ci[1],
        logic50_ci_upper=logic50_ci[2],
        logic50_ci_width=logic50_ci[2] - logic50_ci[1],
        file=row.file
    ))
end # for

## =============================================================================

println("Filter data by IC50 credible region width...")

# Define threshold
thresh = 1.5

# Filter data
df_ic50_thresh = df_ic50[df_ic50.logic50_ci_width.≤thresh, :]

# Group data by :strain_num and :day
df_group = DF.groupby(df_ic50_thresh, [:strain_num, :day])

# Locate groups that include all unique :drug values
group_idx = [
    length(setdiff(df_ic50_thresh.drug, data.drug)) == 0
    for data in df_group
]

# Filter groups
df_group = df_group[group_idx]

## =============================================================================

println("Build logIC50 default matrix...")

# Initialize matrix to save logic50 values
logic50_mean = Matrix{Float32}(
    undef, length(unique(df_ic50_thresh.drug)), length(df_group)
)
logic50_lower = similar(logic50_mean)
logic50_upper = similar(logic50_mean)

# Initialize dataframe to save metadata
df_filt_meta = DF.DataFrame()

# Loop through groups
for (i, data) in enumerate(df_group)
    # Sort data by stress
    DF.sort!(data, :drug)
    if all(data.drug .== sort(unique(df_ic50_thresh.drug)))
        # Add data to matrix
        logic50_mean[:, i] = Float32.(data.logic50_mean)
        logic50_lower[:, i] = Float32.(data.logic50_ci_lower)
        logic50_upper[:, i] = Float32.(data.logic50_ci_upper)
        # Append metadata to dataframe
        DF.append!(df_filt_meta, data[1:1, [:day, :strain_num, :design, :env]])
    else
        println("group $i stress does not match")
    end # if
end # for

# Add strain and evolution condition by splitting env
DF.insertcols!(
    df_filt_meta,
    :strain => getindex.(split.(df_filt_meta.env, "_in_"), 1),
    :evolution => getindex.(split.(df_filt_meta.env, "_in_"), 2)
)

## =============================================================================

println("Build full logIC50 matrix with MCMC posterior samples...")

# Load example MCMC file
mcmc_example = Float32.(Vector(CSV.read(df_ic50.file[1], DF.DataFrame).logic50))

# Initialize array to store all MCMC samples
logic50_mcmc = Array{Float32,3}(
    undef, size(logic50_mean, 1), size(logic50_mean, 2), length(mcmc_example)
)

# Loop through groups
Threads.@threads for i = 1:size(logic50_mean, 2)
    # Extract data
    data = df_group[i]
    # Sorrt data by stress
    DF.sort!(data, :drug)
    # Loop through each drug
    for (j, drug) in enumerate(data.drug)
        # Load MCMC samples
        mcmc = Float32.(Vector(CSV.read(data.file[j], DF.DataFrame).logic50))
        # Store in array
        logic50_mcmc[j, i, :] = mcmc
    end # for
end # for

## =============================================================================

println("Standardize data to mean zero and standard deviation 1 on each environment...")

# Fit model to standardize data to mean zero and standard deviation 1 on each
# environment
dt = StatsBase.fit(StatsBase.ZScoreTransform, logic50_mean, dims=2)

# Center data to have mean zero and standard deviation one
logic50_mean_std = StatsBase.transform(dt, logic50_mean)
logic50_lower_std = StatsBase.transform(dt, logic50_lower)
logic50_upper_std = StatsBase.transform(dt, logic50_upper)

logic50_mcmc_std = reduce(
    (x, y) -> cat(x, y, dims=3),
    StatsBase.transform.(Ref(dt), eachslice(logic50_mcmc, dims=3))
)

## =============================================================================

# Loop through groups to add standardized data to dataframe
for (i, data) in enumerate(df_group)
    # Sort data by stress
    DF.sort!(data, :drug)
    # Add standardized data to dataframe
    data.logic50_mean_std = logic50_mean_std[:, i]
    data.logic50_ci_lower_std = logic50_lower_std[:, i]
    data.logic50_ci_upper_std = logic50_upper_std[:, i]
end # for

# Concatentate all dataframes
df_std = vcat(df_group...)
# Remove file column
df_std = df_std[:, DF.Not(:file)]

## =============================================================================

println("Save objects into JLD2 file...")

JLD2.save(
    "$(git_root())/output/mcmc_iwasawa_logistic/logic50_preprocess.jld2",
    Dict(
        "logic50_mean" => logic50_mean,
        "logic50_mean_std" => logic50_mean_std,
        "logic50_mcmc" => logic50_mcmc,
        "logic50_mcmc_std" => logic50_mcmc_std,
        "logic50_meta" => df_filt_meta,
        "drugs" => sort(unique(df_ic50_thresh.drug))
    )
)

## =============================================================================

println("Save dataframe into CSV files...")

CSV.write(
    "$(git_root())/output/mcmc_iwasawa_logistic/logic50_ci.csv",
    df_std
)