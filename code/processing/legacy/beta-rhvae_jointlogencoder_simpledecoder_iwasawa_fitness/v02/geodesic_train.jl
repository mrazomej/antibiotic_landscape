## =============================================================================
println("Loading packages...")

# Import project package
import AutoEncode

# Import libraries to handel data
import CSV
import DataFrames as DF
import Glob

# Import ML libraries
import Flux

# Import library to save models
import JLD2

# Import basic math
import StatsBase
import Random
Random.seed!(42)

## =============================================================================

# Define model hyperparameters

# Define number of time points to evaluate along curve
n_time = 50
# Define number of epochs
n_epoch = 50_000
# Define learning rate
η = 10^-4

## =============================================================================

println("Loading NeuralGeodesic model...")

# Load model template
nng_template = JLD2.load("./output/geodesic.jld2")["model"]

# Load parameters
Flux.loadmodel!(
    nng_template, JLD2.load("./output/geodesic.jld2")["model_state"]
)

## =============================================================================

println("Loading RHVAE models...")

# List files in output directory
files = Glob.glob("output/model_state/*.jld2")

# Locate unique temperatures in file names
temps = sort(unique([split(f, "_")[end-1] for f in files]))

# Initialize empty dictionary to store models
rhvae_dict = Dict()

# Load model template
rhvae_template = JLD2.load("./output/model.jld2")["model"]

# Loop through temperatures
for (i, temp) in enumerate(temps)
    # Add a dictionary entry for the temperature
    rhvae_dict["$(temp)"] = Dict()
    # Parse temperature value
    T = parse(Float32, replace(temp, "temp" => ""))
    # Load model
    rhvae_dict["$(temp)"]["model"] = AutoEncode.RHVAEs.RHVAE(
        deepcopy(rhvae_template.vae),
        deepcopy(rhvae_template.metric_chain),
        deepcopy(rhvae_template.centroids_data),
        deepcopy(rhvae_template.centroids_latent),
        deepcopy(rhvae_template.L),
        deepcopy(rhvae_template.M),
        T,
        deepcopy(rhvae_template.λ)
    )
    # Search for model files
    model_files = Glob.glob("./output/model_state/*$(temp)*.jld2")
    # Load parameters
    model_state = JLD2.load(model_files[end])["model_state"]
    # Set model parameters
    Flux.loadmodel!(rhvae_dict["$(temp)"]["model"], model_state)
    # Update metric
    AutoEncode.RHVAEs.update_metric!(rhvae_dict["$(temp)"]["model"])
end # for

## =============================================================================

println("Loading data...")

# Define data directory
data_dir = "$(git_root())/data/Iwasawa_2022"

# Load file into memory
df_ic50 = CSV.read("$(data_dir)/iwasawa_ic50_tidy.csv", DF.DataFrame)

# Locate strains with missing values
missing_strains = unique(df_ic50[ismissing.(df_ic50.log2ic50), :strain])

# Remove data
df_ic50 = df_ic50[[x ∉ missing_strains for x in df_ic50.strain], :]

# Group data by strain and day
df_group = DF.groupby(df_ic50, [:strain, :day])

# Extract unique drugs to make sure the matrix is built correctly
drug = sort(unique(df_ic50.drug))

# Initialize matrix to save ic50 values
ic50_mat = Matrix{Float32}(undef, length(drug), length(df_group))

# Loop through groups
for (i, data) in enumerate(df_group)
    # Sort data by stress
    DF.sort!(data, :drug)
    # Check that the stress are in the correct order
    if all(data.drug .== drug)
        # Add data to matrix
        ic50_mat[:, i] = Float32.(data.log2ic50)
    else
        println("group $i stress does not match")
    end # if
end # for

# Define number of environments
n_env = size(ic50_mat, 1)
# Define number of samples
n_samples = size(ic50_mat, 2)

# Fit model to standardize data to mean zero and standard deviation 1 on each
# environment
dt = StatsBase.fit(StatsBase.ZScoreTransform, ic50_mat, dims=2)

# Center data to have mean zero and standard deviation one
ic50_std = StatsBase.transform(dt, ic50_mat)

## =============================================================================

println("Mapping data to latent spaces...")

# Loop through temperatures
for temp in temps
    # Encode data to latent space
    latent = rhvae_dict["$(temp)"]["model"].vae.encoder(ic50_std).µ
    # Add latent space to dictionary
    rhvae_dict["$(temp)"]["latent"] = latent
end # for

## =============================================================================

println("Transforming latent space data to tidy format...")

# Extract strains as ordered in ic50 matrix
strains_mat = [x.strain for x in keys(df_group)]
day_mat = [x.day for x in keys(df_group)]

# Initialize empty dataframe
df_latent = DF.DataFrame()

# Loop through temperatures
for temp in temps
    # Extract latent space
    latent = rhvae_dict["$(temp)"]["latent"]
    # Generate dataframe with corresponding metadata
    DF.append!(
        df_latent,
        DF.DataFrame(
            strain=strains_mat,
            day=day_mat,
            z1=latent[1, :],
            z2=latent[2, :],
            temp=temp
        )
    )
end # for

# Add corresponding metadata resistance value
df_latent = DF.leftjoin!(
    df_latent,
    unique(df_ic50[:, [:strain, :day, :parent, :env]]),
    on=[:strain, :day]
)

## =============================================================================

println("Setting output directories...")

# Define output directory
out_dir = "./output/geodesic_state/"

# Check if output directory exists
if !isdir("./output/")
    mkdir("./output/")
end # if

# Check if output directory exists
if !isdir(out_dir)
    mkdir(out_dir)
end # if

## =============================================================================

# Define time points along curve
t_array = Float32.(collect(range(0, 1, length=n_time)))

# Group latent data by environment, temp, and strain
df_group = DF.groupby(df_latent, [:env, :temp, :strain])

# Define loss function keyword arguments
loss_kwargs = Dict(
    :curve_velocity => NG.curve_velocity_finitediff,
    :curve_integral => NG.curve_energy,
)

# Loop through groups
Threads.@threads for i in eachindex(df_group)
    # Extract data and sort by day
    data = DF.sort!(df_group[i], :day)
    # Extract group metadata
    env = data.env[1]
    temp = data.temp[1]
    strain = data.strain[1]
    println("Training $(temp) | $(env) | $(strain)")
    # Extract initial and final points for strain
    z_init = [data.z1[1], data.z2[1]]
    z_end = [data.z1[end], data.z2[end]]
    # Set NeuralGeodesic model
    nng = NG.NeuralGeodesic(
        deepcopy(nng_template.mlp),
        z_init,
        z_end
    )
    # Explicit setup of optimizer
    opt_nng = Flux.Train.setup(
        Flux.Optimisers.Adam(η),
        nng
    )
    # Initialize empty array to save loss
    nng_loss = Vector{Float32}(undef, n_epoch)
    # Loop through epochs
    for epoch in 1:n_epoch
        # Train model and save loss
        nng_loss[epoch] = NG.train!(
            nng,
            deepcopy(rhvae_dict["$(temp)"]["model"]),
            t_array,
            opt_nng;
            loss_kwargs=loss_kwargs, loss_return=true
        )
    end # for

    # Save checkpoint
    JLD2.jldsave(
        "$(out_dir)/neuralgeodesic_$(temp)_$(env)env_" *
        "$(replace(strain, " " => "-"))strain" *
        "_$(lpad(n_epoch, 5, "0"))epoch.jld2",
        model_state=Flux.state(nng),
        z_init=z_init,
        z_end=z_end,
        env=env,
        temp=temp,
        strain=strain,
    )
end # for