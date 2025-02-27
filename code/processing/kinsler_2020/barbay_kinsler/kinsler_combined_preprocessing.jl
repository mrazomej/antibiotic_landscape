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

data_dir = "$(git_root())/output/barbay_kinsler_2020/"

println("Listing files for hierarchical inference...")


# Define data directory for hierarchical inference
data_hier_dir = "$(git_root())/output/barbay_kinsler_2020/" *
                "advi_meanfield_hierarchicalreplicate_inference"

# List all files in the data directory
files = sort(Glob.glob("$(data_hier_dir)/advi*.csv"[2:end], "/"))

# Initialize empty dataframe to store metadata
df_meta = DF.DataFrame()

# Loop over each file and extract metadata
for file in files
    # Extract filename
    fname = split(file, "/")[end]

    # Extract environment from filename using regular expressions
    env = match(r"_([^_]+)env", fname).captures[1]

    # Extract number of samples during ADVI inference using regular expressions
    n_samples = parse(Int, match(r"_(\d+)samples", fname).captures[1])

    # Extract number of steps during ADVI inference using regular expressions
    n_steps = parse(Int, match(r"_(\d+)steps", fname).captures[1])

    # Create a new row with the extracted metadata
    DF.append!(df_meta, DF.DataFrame(
        env=env,
        n_samples=n_samples,
        n_steps=n_steps,
        file=file,
        type="hierarchical",
        rep="R1"
    ))
end

println("Number of datasets: $(size(df_meta, 1))")

## =============================================================================

println("Listing files for non-hierarchical inference...")

# Define data directory for non-hierarchical inference
data_nonhier_dir = "$(git_root())/output/barbay_kinsler_2020/" *
                   "advi_meanfield_joint_inference"


# List all files in the data directory
files = sort(Glob.glob("$(data_nonhier_dir)/kinsler*.csv"[2:end], "/"))

# Loop over each file and extract metadata
for file in files
    # Extract filename
    fname = split(file, "/")[end]

    # Extract environment from filename using regular expressions
    env = match(r"_([^_]+)env", fname).captures[1]

    # Extract number of samples during ADVI inference using regular expressions
    n_samples = parse(Int, match(r"_(\d+)samples", fname).captures[1])

    # Extract number of steps during ADVI inference using regular expressions
    n_steps = parse(Int, match(r"_(\d+)steps", fname).captures[1])

    # Extract replicate number from filename using regular expressions
    rep = match(r"_([^_]+)rep", fname).captures[1]

    # Create a new row with the extracted metadata
    DF.append!(df_meta, DF.DataFrame(
        env=env,
        n_samples=n_samples,
        n_steps=n_steps,
        file=file,
        type="joint",
        rep=rep
    ))
end

println("Number of datasets: $(size(df_meta, 1))")

## =============================================================================

# Find environments in both hierarchical and non-hierarchical inference
envs_hier = unique(df_meta[df_meta.type.=="hierarchical", :env])
envs_nonhier = unique(df_meta[df_meta.type.=="joint", :env])

# Find environments in both hierarchical and non-hierarchical inference
envs_both = intersect(envs_hier, envs_nonhier)

# Find environments only in non-hierarchical inference
envs_only_nonhier = setdiff(envs_nonhier, envs_hier)

println("Number of environments in both hierarchical and " *
        "non-hierarchical inference: $(length(envs_both))")
println("Number of environments only in non-hierarchical inference: " *
        "$(length(envs_only_nonhier))")

# Extract data type="joint", and env in envs_only_nonhier
df_joint = df_meta[(df_meta.type.=="joint").&(in.(df_meta.env, Ref(envs_only_nonhier))), :]

println("Number of datasets in joint inference: $(size(df_joint, 1))")

## =============================================================================

println("Loading hierarchical data...")

# Initialize dataframe to agglomerate all results
df_kinsler_hier = DF.DataFrame()

# Loop over each file
for data in eachrow(df_meta[df_meta.type.=="hierarchical", :])
    # Read the file
    df_param = CSV.read(data.file, DF.DataFrame)

    # Filter hyperfitness parameters
    df_env = df_param[
        df_param.vartype.=="bc_hyperfitness", [:mean, :std, :id]]

    # Add environment to the dataframe
    df_env.env .= data.env
    # Add replicate to the dataframe
    df_env.rep .= data.rep
    # Add type to the dataframe
    df_env.type .= data.type

    # Append to the dataframe
    DF.append!(df_kinsler_hier, df_env)
end # for

# Rename columns
DF.rename!(df_kinsler_hier, :mean => :fitness_mean, :std => :fitness_std)

## =============================================================================

println("Loading joint data...")

# Initialize dataframe to agglomerate all results
df_kinsler_joint = DF.DataFrame()

# Loop over each file
for data in eachrow(df_joint)
    # Read the file
    df_param = CSV.read(data.file, DF.DataFrame)

    # Filter hyperfitness parameters
    df_env = df_param[df_param.vartype.=="bc_fitness", [:mean, :std, :id]]

    # Add environment to the dataframe
    df_env.env .= data.env
    # Add replicate to the dataframe
    df_env.rep .= data.rep
    # Add type to the dataframe
    df_env.type .= data.type

    # Append to the dataframe
    DF.append!(df_kinsler_joint, df_env)
end # for

# Rename columns
DF.rename!(df_kinsler_joint, :mean => :fitness_mean, :std => :fitness_std)

## =============================================================================

println("Concatenating hierarchical and joint dataframes...")

# Concatenate hierarchical and joint dataframes
df_kinsler = DF.append!(df_kinsler_hier, df_kinsler_joint)

## =============================================================================

println("Checking if all environments contain the same set of IDs...")

# Group data by environment
df_group = DF.groupby(df_kinsler, :env)

# Get set of IDs for each environment
env_ids = Dict(
    name => Set(group.id)
    for (name, group) in pairs(df_group)
)

# Check if all environments have the same IDs
all_equal = all(ids -> ids == first(values(env_ids)), values(env_ids))

if all_equal
    println("All environments contain the same set of IDs")
else
    # Print differences if they exist
    println("Environments have different IDs:")
    for (env1, ids1) in env_ids
        for (env2, ids2) in env_ids
            if env1 < env2  # avoid printing both (A,B) and (B,A)
                unique_to_1 = setdiff(ids1, ids2)
                unique_to_2 = setdiff(ids2, ids1)
                if !isempty(unique_to_1) || !isempty(unique_to_2)
                    println("  $env1 vs $env2:")
                    !isempty(unique_to_1) && println("    Only in $env1: ", unique_to_1)
                    !isempty(unique_to_2) && println("    Only in $env2: ", unique_to_2)
                end # if
            end # for
        end # for
    end # for
end # for

## =============================================================================

println(
    "Standardizing fitness values to mean zero and standard deviation one..."
)

# Pivot dataframe
df_pivot_mean = DF.unstack(df_kinsler, :env, :id, :fitness_mean)
df_pivot_std = DF.unstack(df_kinsler, :env, :id, :fitness_std)

# Extract fitness values as matrix
fitness_matrix_mean = Matrix(Float64.(df_pivot_mean[:, DF.Not(:env)]))
fitness_matrix_std = Matrix(Float64.(df_pivot_std[:, DF.Not(:env)]))

# Fit standardization to mean
dt = StatsBase.fit(StatsBase.ZScoreTransform, fitness_matrix_mean, dims=2)

# Center data to have mean zero and standard deviation one
fitness_matrix_mean_std = StatsBase.transform(dt, fitness_matrix_mean)

# Fit standardization to mean, but do not center
dt_std = StatsBase.fit(
    StatsBase.ZScoreTransform,
    fitness_matrix_mean,
    dims=2,
    center=false
)

# Standardize standard deviation to have mean zero and standard deviation one
fitness_matrix_std_std = StatsBase.transform(dt_std, fitness_matrix_std)

# Convert to dataframe
df_kinsler_mean_std = DF.DataFrame(
    fitness_matrix_mean_std, names(df_pivot_mean)[2:end]
)
df_kinsler_std_std = DF.DataFrame(
    fitness_matrix_std_std, names(df_pivot_std)[2:end]
)

# Add environment to the dataframe
df_kinsler_mean_std.env .= df_pivot_mean.env
df_kinsler_std_std.env .= df_pivot_std.env

# Convert to tidy format
df_kinsler_mean_std = DF.stack(
    df_kinsler_mean_std, DF.Not(:env),
    value_name=:fitness_mean_standard,
    variable_name=:id
)
df_kinsler_std_std = DF.stack(
    df_kinsler_std_std, DF.Not(:env),
    value_name=:fitness_std_standard,
    variable_name=:id
)

# Combine dataframes
df_kinsler_standard = DF.leftjoin(
    df_kinsler_mean_std, df_kinsler_std_std, on=[:id, :env]
)

## =============================================================================

# Combine dataframes
df_kinsler = DF.leftjoin(df_kinsler, df_kinsler_standard, on=[:id, :env])

## =============================================================================

# Read dataframe with id metadata
df_meta = CSV.read(
    "$(git_root())/data/Kinsler_2020/tidy_counts.csv",
    DF.DataFrame
)

# List relevant columns
cols = [
    :additional_muts,
    :barcode,
    :class,
    :gene,
    :ploidy,
    :type,
]

# Extract relevant columns
df_meta = unique(df_meta[:, cols])

# Rename :barcode => :id, :type => :mut_type
DF.rename!(df_meta, :barcode => :id, :type => :mut_type)

# Convert :id column to string
df_meta.id = string.(df_meta.id)

# Keep only ids that are in df_kinsler
df_meta = df_meta[in.(df_meta.id, Ref(df_kinsler.id)), :]

# Combine dataframes
df_kinsler_meta = DF.leftjoin(df_kinsler, df_meta, on=:id)

## =============================================================================

println("Saving data...")

# Save the dataframe
CSV.write("$(data_dir)/kinsler_combined_hyperfitness.csv", df_kinsler)