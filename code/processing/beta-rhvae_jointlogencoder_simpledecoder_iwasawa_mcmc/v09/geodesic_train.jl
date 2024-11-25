## =============================================================================
println("Loading packages...")

# Import project package
import AutoEncoderToolkit as AET
import AutoEncoderToolkit.diffgeo.NeuralGeodesics as NG

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

# Locate current directory
path_dir = pwd()

# Define data directory
data_dir = "$(git_root())/output/mcmc_iwasawa_logistic"

# Find the path prefix where to store output
out_prefix = replace(
    match(r"processing/(.*)", path_dir).match, "processing" => ""
)

# Define output directory
out_dir = "$(git_root())/output$(out_prefix)"

# Define model directory
model_dir = "$(git_root())/output$(out_prefix)/model_state"

# Define directory to store trained geodesic curves
geodesic_dir = "$(git_root())/output$(out_prefix)/geodesic_state/"

# Generate output directory if it doesn't exist
if !isdir(geodesic_dir)
    println("Generating output directory...")
    mkpath(geodesic_dir)
end

## =============================================================================

println("Loading NeuralGeodesic model...")

# Load model template
nng_template = JLD2.load("$(out_dir)/geodesic.jld2")["model"]

# Load parameters
Flux.loadmodel!(
    nng_template, JLD2.load("$(out_dir)/geodesic.jld2")["model_state"]
)

# Define time points along curve
t_array = Float32.(collect(range(0, 1, length=n_time)))

## =============================================================================

println("Loading RHVAE model...")

# Load RHVAE model
rhvae = JLD2.load("$(out_dir)/model.jld2")["model"]
# List parameters for epochs
param_files = sort(Glob.glob("$(model_dir)/*.jld2"[2:end], "/"))
# Load last epoch
Flux.loadmodel!(rhvae, JLD2.load(param_files[end])["model_state"])
# Update metric
AET.RHVAEs.update_metric!(rhvae)

## =============================================================================

println("Loading IC50 data...")

df_logic50 = CSV.read("$(data_dir)/logic50_ci.csv", DF.DataFrame)

println("Map data to latent space...")

# Group dataframe by :day, :strain_num, and :env
df_group = DF.groupby(df_logic50, [:day, :strain_num, :env])
# Initialize empty dataframe to store latent coordinates
df_latent = DF.DataFrame()
# Loop over groups
for data in df_group
    # Sort data by drug
    DF.sort!(data, :drug)
    # Run :logic50_mean_std through encoder
    latent = rhvae.vae.encoder(data.logic50_mean_std).µ
    # Append latent coordinates to dataframe
    DF.append!(
        df_latent,
        DF.DataFrame(
            :day .=> first(data.day),
            :strain_num .=> first(data.strain_num),
            :meta .=> first(data.env),
            :env .=> split(first(data.env), "_")[end],
            :strain .=> split(first(data.env), "_")[1],
            :latent1 => latent[1, :],
            :latent2 => latent[2, :],
            :latent3 => latent[3, :],
        )
    )
end # for 

## =============================================================================

println("Define lineages over which to train geodesic...")

# Group data by :env and :strain_num
df_group = DF.groupby(df_latent, [:env, :strain_num])

# Initialize empty dataframe to store data from IDs to be selected for geodesic
# analysis
df_geodesic = DF.DataFrame()

# Loop through each group
Threads.@threads for i in 1:length(df_group)
    # Extract data
    data = df_group[i]
    # Sort data by day
    DF.sort!(data, :day)
    # Define metadata
    env = first(data.env)
    strain_num = first(data.strain_num)
    parent = first(data.strain)
    println("Processing lineage: $(i)/$(length(df_group)) | $(env) | $(strain_num) | $(parent)")
    # Find minimum day
    day_init = minimum(data.day)
    # Find maximum day 
    day_final = maximum(data.day)
    # Define output name
    fname = "$(geodesic_dir)/neuralgeodesic_" *
            "dayinit$(lpad(day_init, 2, "0"))_" *
            "dayfinal$(lpad(day_final, 2, "0"))_" *
            "evoenv$(env)_" *
            "id$(lpad(strain_num, 3, "0"))_" *
            "rhvaeepoch$(lpad(length(param_files), 4, "0"))_" *
            "geoepoch$(lpad(n_epoch, 4, "0")).jld2"

    # Check if file exists
    if isfile(fname)
        continue
    end # if 

    # Extract initial and final points in latent space
    latent_init = Float32.(Array(data[1, [:latent1, :latent2, :latent3]]))
    latent_end = Float32.(Array(data[end, [:latent1, :latent2, :latent3]]))
    # Set NeuralGeodesic model
    nng = NG.NeuralGeodesic(
        deepcopy(nng_template.mlp),
        latent_init,
        latent_end
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
        if epoch % 2_500 == 0
            println("$(env) | $(strain_num) | $(parent)")
            println("   - Epoch: $(epoch)")
        end
        # Train model and save loss
        nng_loss[epoch] = NG.train!(nng, rhvae, t_array, opt_nng; loss_return=true)
    end # for
    # Save network
    JLD2.jldsave(
        fname,
        model_state=Flux.state(nng),
        latent_init=latent_init,
        latent_end=latent_end,
    )
end # for data in data_group
