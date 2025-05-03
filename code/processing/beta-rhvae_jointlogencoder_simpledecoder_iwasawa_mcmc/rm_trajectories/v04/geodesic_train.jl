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

# Find model file
model_file = first(Glob.glob("$(out_dir)/model*.jld2"[2:end], "/"))
# List epoch parameters
model_states = Glob.glob("$(model_dir)/*.jld2"[2:end], "/")

# Initialize dataframe to store files metadata
df_meta = DF.DataFrame()

# Loop over files
for f in model_states
    # Extract epoch number from file name using regular expression
    epoch = parse(Int, match(r"epoch(\d+)", f).captures[1])
    # Extract evolution condition removed
    evo = match(r"rhvae_([^_]+)_epoch", f).captures[1]
    # Load model_state file
    f_load = JLD2.load(f)
    # Extract values
    loss_train = f_load["loss_train"]
    loss_val = f_load["loss_val"]
    mse_train = f_load["mse_train"]
    mse_val = f_load["mse_val"]
    # Generate temporary dataframe to store metadata
    df_tmp = DF.DataFrame(
        :epoch => epoch,
        :evo => evo,
        :loss_train => loss_train,
        :loss_val => loss_val,
        :mse_train => mse_train,
        :mse_val => mse_val,
        :model_file => model_file,
        :model_state => f,
    )
    # Append temporary dataframe to main dataframe
    global df_meta = DF.vcat(df_meta, df_tmp)
end # for f in model_states
## =============================================================================

println("Load models...")

# Initialize dictionary to store models
rhvae_models = Dict()

# Loop through each unique evolution condition
for evo in unique(df_meta.evo)
    # Get the latest model state file for this evolution condition
    latest_model_state = df_meta[df_meta.evo.==evo, :model_state][end]
    # Load base model
    rhvae = JLD2.load(model_file)["model"]
    # Load latest model state for this evolution condition
    Flux.loadmodel!(rhvae, JLD2.load(latest_model_state)["model_state"])
    # Update metric parameters
    AET.RHVAEs.update_metric!(rhvae)
    # Store the model in the dictionary
    rhvae_models[evo] = rhvae
end

## =============================================================================

println("Loading IC50 data...")

# Load logic50 data 
df_logic50 = CSV.read("$(data_dir)/logic50_ci.csv", DF.DataFrame)
# Extract strain and evolution condition from :env by splitting by _
DF.insertcols!(
    df_logic50,
    :strain => getindex.(split.(df_logic50.env, "_"), 1),
    :evo => getindex.(split.(df_logic50.env, "_"), 3),
)

## =============================================================================

println("Map data to latent space...")

# Group dataframe by :day, :strain_num, and :env
df_group = DF.groupby(df_logic50, [:day, :strain_num, :env])
# Initialize empty dataframe to store latent coordinates
df_latent = DF.DataFrame()
# Loop over groups
for (i, data) in enumerate(df_group)
    # Sort data by drug
    DF.sort!(data, :drug)
    # Loop over each model (one for each evolution condition)
    for (evo, rhvae) in rhvae_models
        # Run :logic50_mean_std through encoder
        latent = rhvae.vae.encoder(data.logic50_mean_std).µ
        # Determine if data is in training or validation set for this model
        train = evo ≠ first(data.evo)
        # Append latent coordinates to dataframe
        DF.append!(
            df_latent,
            DF.DataFrame(
                :day .=> first(data.day),
                :strain_num .=> first(data.strain_num),
                :meta .=> first(data.env),
                :evo .=> first(data.evo),
                :strain .=> first(data.strain),
                :latent1 => latent[1, :],
                :latent2 => latent[2, :],
                :train => train,
                :model_evo => evo
            )
        )
    end # for rhvae_models
end # for df_group

## =============================================================================

println("Define lineages over which to train geodesic...")

# Group data by :evo, :strain_num, and :model_evo
df_group = DF.groupby(df_latent, [:evo, :strain_num, :model_evo])

# Initialize empty dataframe to store data from IDs to be selected for geodesic
# analysis
df_geodesic = DF.DataFrame()

# Loop through each group
Threads.@threads for i in 1:length(df_group)
    data = df_group[i]
    # Sort data by day
    DF.sort!(data, :day)
    # Define metadata
    evo = first(data.evo)
    strain_num = first(data.strain_num)
    parent = first(data.strain)
    model_evo = first(data.model_evo)
    println("Processing lineage: $(i)/$(length(df_group)) | $(evo) | $(strain_num) | $(parent) | Model: $(model_evo)")
    # Find minimum day
    day_init = minimum(data.day)
    # Find maximum day 
    day_final = maximum(data.day)
    # Define output name
    fname = "$(geodesic_dir)/neuralgeodesic_" *
            "dayinit$(lpad(day_init, 2, "0"))_" *
            "dayfinal$(lpad(day_final, 2, "0"))_" *
            "evoenv$(evo)_" *
            "id$(lpad(strain_num, 3, "0"))_" *
            "rhvaemodel$(model_evo)_" *
            "rhvaeepoch$(lpad(length(model_states), 4, "0"))_" *
            "geoepoch$(lpad(n_epoch, 4, "0")).jld2"

    # Check if file exists
    if isfile(fname)
        continue
    end # if 

    # Extract initial and final points in latent space
    latent_init = Float32.(Array(data[1, [:latent1, :latent2]]))
    latent_end = Float32.(Array(data[end, [:latent1, :latent2]]))
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
    # Get the corresponding RHVAE model
    rhvae = rhvae_models[model_evo]
    # Loop through epochs
    for epoch in 1:n_epoch
        if epoch % 2_500 == 0
            println("       - Epoch: $(epoch)")
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
        nng_loss=nng_loss,
    )
end # for data in data_group

## =============================================================================