## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic

# Import AutoEncoderToolkit to train VAEs
import AutoEncoderToolkit as AET

# Import libraries to handle data
import Glob
import DataFrames as DF
import CSV

# Import ML libraries
import Flux

# Import library to save models
import JLD2

# Import IterTools for Cartesian product
import IterTools

# Import basic math
import LinearAlgebra
import StatsBase
import Random

# Load Plotting packages
using CairoMakie
using Makie
import ColorSchemes
import Colors
# Activate backend
CairoMakie.activate!()

# Set plotting style
Antibiotic.viz.theme_makie!()

# Set random seed
Random.seed!(42)

## =============================================================================

println("Defining directories...")

# Define path prefix
path_prefix = "beta-rhvae_jointlogencoder_simpledecoder_iwasawa_mcmc/" *
              "rm_antibiotic/v01"


# Define data directory
data_dir = "$(git_root())/output/mcmc_iwasawa_logistic"
# Define output directory
out_dir = "$(git_root())/output/$(path_prefix)"
# Define RHVAE model state directory
rhvae_state_dir = "$(git_root())/output/$(path_prefix)/rhvae_model_state"
# Define RHVAE crossvalidation directory
rhvae_crossval_dir = "$(git_root())/output/$(path_prefix)/rhvae_crossvalidation_state"
# Define VAEs model state directory
vae_state_dir = "$(git_root())/output/$(path_prefix)/vae_model_state"
# Define VAEs crossvalidation directory
vae_crossval_dir = "$(git_root())/output/$(path_prefix)/vae_crossvalidation_state"

# Define figure directory
fig_dir = "$(git_root())/fig/main"

# Create figure directory if it does not exist
if !isdir(fig_dir)
    println("Creating figure directory...")
    mkpath(fig_dir)
end

## =============================================================================

println("Loading data into memory...")

# Load standardized mean data
logic50_mean = JLD2.load(
    "$(data_dir)/logic50_preprocess.jld2"
)["logic50_mean_std"]
# Load data metadata
data_meta = JLD2.load(
    "$(data_dir)/logic50_preprocess.jld2"
)["logic50_meta"]
# Load drug list
drug_list = JLD2.load(
    "$(data_dir)/logic50_preprocess.jld2"
)["drugs"]
# Load IC50 data
df_logic50 = CSV.read("$(data_dir)/logic50_ci.csv", DF.DataFrame)
# Load column metadata
df_col_meta = JLD2.load(
    "$(data_dir)/logic50_preprocess.jld2"
)["logic50_meta"]

## =============================================================================

println("Loading errors...")

# List RHVAE model states
rhvae_model_states = sort(
    Glob.glob("$(rhvae_state_dir)/beta-rhvae_*_epoch*.jld2"[2:end], "/")
)

# List RHVAE crossvalidation states
rhvae_crossval_states = sort(
    Glob.glob("$(rhvae_crossval_dir)/beta-rhvae_*_crossval*_epoch*.jld2"[2:end], "/")
)

# List VAE model states
vae_model_states = sort(
    Glob.glob("$(vae_state_dir)/beta-vae_*_epoch*.jld2"[2:end], "/")
)

# List VAE crossvalidation states
vae_crossval_states = sort(
    Glob.glob("$(vae_crossval_dir)/beta-vae_*_crossval*_epoch*.jld2"[2:end], "/")
)

# Initialize dataframe to store files metadata
df_meta = DF.DataFrame()

# Loop over RHVAE model states
for file in rhvae_model_states
    # Extract epoch number from file name using regular expression
    epoch = parse(Int, match(r"epoch(\d+)", file).captures[1])
    # Extract drug name from file name using regular expression
    drug = match(r"beta-rhvae_(.+?)rm_epoch", split(file, "/")[end]).captures[1]
    # Load model_state file
    model_load = JLD2.load(file)
    # Extract loss train and loss val
    loss_train = model_load["loss_train"]
    loss_val = model_load["loss_val"]
    # Extract mse train and mse val
    mse_train = model_load["mse_train"]
    mse_val = model_load["mse_val"]
    # Generate temporary dataframe to store metadata
    df_tmp = DF.DataFrame(
        :epoch => epoch,
        :drug => drug,
        :loss_train => loss_train,
        :loss_val => loss_val,
        :mse_train => mse_train,
        :mse_val => mse_val,
        :model => "rhvae",
        :model_file => file,
        :model_state => file,
        :model_type => "original",
        :train_frac => 0.85,
    )
    # Append temporary dataframe to main dataframe
    global df_meta = DF.vcat(df_meta, df_tmp)
end # for RHVAE model_state

# Loop over RHVAE crossvalidation states
for file in rhvae_crossval_states
    # Extract epoch number from file name using regular expression
    epoch = parse(Int, match(r"epoch(\d+)", file).captures[1])
    # Extract drug name from file name using regular expression
    drug = match(r"beta-rhvae_(.+?)rm_crossval", split(file, "/")[end]).captures[1]
    # Extract split fraction from file name using regular expression
    split_frac = parse(Float64, match(r"(\d+\.\d+)split", file).captures[1])
    # Load model_state file
    model_load = JLD2.load(file)
    # Extract loss train and loss val
    loss_train = model_load["loss_train"]
    loss_val = model_load["loss_val"]
    # Extract mse train and mse val
    mse_train = model_load["mse_train"]
    mse_val = model_load["mse_val"]
    # Generate temporary dataframe to store metadata
    df_tmp = DF.DataFrame(
        :epoch => epoch,
        :drug => drug,
        :loss_train => loss_train,
        :loss_val => loss_val,
        :mse_train => mse_train,
        :mse_val => mse_val,
        :model => "rhvae",
        :model_file => file,
        :model_state => file,
        :model_type => "crossvalidation",
        :train_frac => split_frac,
    )
    # Append temporary dataframe to main dataframe
    global df_meta = DF.vcat(df_meta, df_tmp)
end # for RHVAE crossvalidation states

# Loop over VAE model states
for file in vae_model_states
    # Extract epoch number from file name using regular expression
    epoch = parse(Int, match(r"epoch(\d+)", file).captures[1])
    # Extract drug name from file name using regular expression
    drug = match(r"vae_(.+?)rm_epoch", split(file, "/")[end]).captures[1]
    # Load model_state file
    model_load = JLD2.load(file)
    # Extract loss train and loss val
    loss_train = model_load["loss_train"]
    loss_val = model_load["loss_val"]
    # Extract mse train and mse val
    mse_train = model_load["mse_train"]
    mse_val = model_load["mse_val"]
    # Generate temporary dataframe to store metadata
    df_tmp = DF.DataFrame(
        :epoch => epoch,
        :drug => drug,
        :loss_train => loss_train,
        :loss_val => loss_val,
        :mse_train => mse_train,
        :mse_val => mse_val,
        :model => "vae",
        :model_file => file,
        :model_state => file,
        :model_type => "original",
        :train_frac => 0.85,
    )
    # Append temporary dataframe to main dataframe
    global df_meta = DF.vcat(df_meta, df_tmp)
end # for VAE model_state

# Loop over VAE crossvalidation states
for file in vae_crossval_states
    # Extract epoch number from file name using regular expression
    epoch = parse(Int, match(r"epoch(\d+)", file).captures[1])
    # Extract drug name from file name using regular expression
    drug = match(r"vae_(.+?)rm_crossval", split(file, "/")[end]).captures[1]
    # Extract split fraction from file name using regular expression
    split_frac = parse(Float64, match(r"(\d+\.\d+)split", file).captures[1])
    # Load model_state file
    model_load = JLD2.load(file)
    # Extract loss train and loss val
    loss_train = model_load["loss_train"]
    loss_val = model_load["loss_val"]
    # Extract mse train and mse val
    mse_train = model_load["mse_train"]
    mse_val = model_load["mse_val"]
    # Generate temporary dataframe to store metadata
    df_tmp = DF.DataFrame(
        :epoch => epoch,
        :drug => drug,
        :loss_train => loss_train,
        :loss_val => loss_val,
        :mse_train => mse_train,
        :mse_val => mse_val,
        :model => "vae",
        :model_file => file,
        :model_state => file,
        :model_type => "crossvalidation",
        :train_frac => split_frac,
    )
    # Append temporary dataframe to main dataframe
    global df_meta = DF.vcat(df_meta, df_tmp)
end # for VAE crossvalidation states

## =============================================================================

println("Loading RHVAE and VAE models...")

# Group data by drug
df_group = DF.groupby(df_meta, [:drug])

# Initialize dictionary to store models
rhvae_dict = Dict{String,Any}()
vae_dict = Dict{String,Any}()

# Loop over drugs
for (i, data) in enumerate(df_group)
    # Extract drug name
    drug = data.drug[1]

    # Filter data by model type
    rhvae_data = data[data.model.=="rhvae", :]
    vae_data = data[data.model.=="vae", :]

    # Process RHVAE models
    if !isempty(rhvae_data)
        # Extract original model
        data_full = DF.sort(rhvae_data[rhvae_data.model_type.=="original", :], [:epoch])
        # Extract crossvalidation model
        data_crossval = DF.sort(
            rhvae_data[rhvae_data.model_type.=="crossvalidation", :], [:epoch]
        )
        # Load model   
        rhvae_full = JLD2.load("$(out_dir)/model_$(drug)rm.jld2")["model"]
        # Load model state
        model_state = JLD2.load(data_full.model_state[end])["model_state"]
        # Input parameters to model
        Flux.loadmodel!(rhvae_full, model_state)
        # Update metric parameters
        AET.RHVAEs.update_metric!(rhvae_full)
        # Load second decoder
        decoder_missing = JLD2.load("$(out_dir)/decoder_missing.jld2")["model"]
        # Build second rhvae
        rhvae_crossval = AET.RHVAEs.RHVAE(
            deepcopy(rhvae_full.vae.encoder) * decoder_missing,
            deepcopy(rhvae_full.metric_chain),
            deepcopy(rhvae_full.centroids_data),
            deepcopy(rhvae_full.T),
            deepcopy(rhvae_full.λ)
        )
        # Load model state
        model_state = JLD2.load(data_crossval.model_state[end])["model_state"]
        # Input parameters to model
        Flux.loadmodel!(rhvae_crossval, model_state)
        # Update metric parameters
        AET.RHVAEs.update_metric!(rhvae_crossval)
        # Load train and validation idx
        full_train_idx = JLD2.load(data_full.model_state[end])["train_idx"]
        full_val_idx = JLD2.load(data_full.model_state[end])["val_idx"]
        crossval_train_idx = JLD2.load(data_crossval.model_state[end])["train_idx"]
        crossval_val_idx = JLD2.load(data_crossval.model_state[end])["val_idx"]
        # Store models
        rhvae_dict[drug] = (
            full=rhvae_full,
            crossval=rhvae_crossval,
            full_train_idx=full_train_idx,
            full_val_idx=full_val_idx,
            crossval_train_idx=crossval_train_idx,
            crossval_val_idx=crossval_val_idx,
        )
    end

    # Process VAE models
    if !isempty(vae_data)
        # Extract original model
        data_full = DF.sort(vae_data[vae_data.model_type.=="original", :], [:epoch])
        # Extract crossvalidation model
        data_crossval = DF.sort(
            vae_data[vae_data.model_type.=="crossvalidation", :], [:epoch]
        )
        # Load model   
        vae_full = JLD2.load("$(out_dir)/vae_model_base.jld2")["model"]
        # Load model state
        model_state = JLD2.load(data_full.model_state[end])["model_state"]
        # Input parameters to model
        Flux.loadmodel!(vae_full, model_state)
        # Load second decoder
        decoder_missing = JLD2.load("$(out_dir)/decoder_missing.jld2")["model"]
        # Build second vae
        vae_crossval = deepcopy(vae_full.encoder) * decoder_missing
        # Load model state
        model_state = JLD2.load(data_crossval.model_state[end])["model_state"]
        # Input parameters to model
        Flux.loadmodel!(vae_crossval, model_state)
        # Load train and validation idx
        full_train_idx = JLD2.load(data_full.model_state[end])["train_idx"]
        full_val_idx = JLD2.load(data_full.model_state[end])["val_idx"]
        crossval_train_idx = JLD2.load(data_crossval.model_state[end])["train_idx"]
        crossval_val_idx = JLD2.load(data_crossval.model_state[end])["val_idx"]
        # Store models
        vae_dict[drug] = (
            full=vae_full,
            crossval=vae_crossval,
            full_train_idx=full_train_idx,
            full_val_idx=full_val_idx,
            crossval_train_idx=crossval_train_idx,
            crossval_val_idx=crossval_val_idx,
        )
    end
end # for drug

## =============================================================================

println("Performing SVD cross validation...")

# Initialize dataframe to store results
df_svd = DF.DataFrame()

# Initialize dictionary to store predictions
svd_pred = Dict{String,Any}()

# Loop through drugs
for (i, drug) in enumerate(drug_list)
    # Extract train and validation idx
    train_idx = rhvae_dict[drug].crossval_train_idx
    val_idx = rhvae_dict[drug].crossval_val_idx
    # Extract environment idx
    env_idx = vcat([i], setdiff(1:length(drug_list), i))
    # Arraynge logic50_mean with these indices
    logic50_svd = Float64.(logic50_mean[env_idx, vcat(val_idx, train_idx)])
    #  Perform svd cross validation
    r2_values, mse_values, predictions = Antibiotic.stats.svd_cross_validation(
        logic50_svd,
        1,
        length(val_idx),
        0
    )
    # Append results to dataframe
    DF.append!(
        df_svd,
        DF.DataFrame(
            :drug .=> drug,
            :r2 => r2_values,
            :mse => mse_values,
            :rank => 1:length(r2_values),
        )
    )
    # Append predictions to dictionary
    svd_pred[drug] = predictions
end # for drug

## =============================================================================

println("Map data to latent space...")

# Group dataframe by :day, :strain_num, and :env
df_group = DF.groupby(df_logic50, [:day, :strain_num, :env])
# Initialize empty dataframe to store latent coordinates
df_latent = DF.DataFrame()
# Loop over groups
for (j, data) in enumerate(df_group)
    # Sort data by drug
    DF.sort!(data, :drug)
    # Loop through drugs
    for (i, drug) in enumerate(data.drug)
        # Find number of row in df_meta with same :strain_num, :day, and :env
        meta_idx = findfirst(
            (df_col_meta.strain_num .== first(data.strain_num)) .&
            (df_col_meta.day .== first(data.day)) .&
            (df_col_meta.env .== first(data.env))
        )
        # Find whether data is train or validation
        full_type = meta_idx in rhvae_dict[drug].full_train_idx ? "train" : "validation"
        crossval_type = meta_idx in rhvae_dict[drug].crossval_train_idx ? "train" : "validation"
        # Define index of drugs to be used
        env_idx = sort(setdiff(1:length(data.drug), i))
        # Extract data to map to latent space
        logic50_std = Float32.(data.logic50_mean_std)
        # Select data to map to latent space
        logic50_full_std = logic50_std[env_idx, :]
        logic50_crossval_std = logic50_std[i, :]

        # Map to RHVAE latent space
        latent_full_rhvae = rhvae_dict[drug].full(logic50_full_std; latent=true)
        # Extract latent coordinates
        latent_rhvae = latent_full_rhvae.phase_space.z_final
        # Reconstruct logic50_mean_std
        logic50_full_recon_rhvae = rhvae_dict[drug].full.vae.decoder(latent_rhvae).µ
        logic50_crossval_recon_rhvae = rhvae_dict[drug].crossval.vae.decoder(latent_rhvae).µ
        # Compute reconstruction error
        logic50_full_mse_rhvae = Flux.mse(logic50_full_std, logic50_full_recon_rhvae)
        logic50_crossval_mse_rhvae = Flux.mse(logic50_crossval_std, logic50_crossval_recon_rhvae)

        # Append RHVAE latent coordinates to dataframe
        DF.append!(
            df_latent,
            DF.DataFrame(
                :day .=> first(data.day),
                :strain_num .=> first(data.strain_num),
                :meta .=> first(data.env),
                :env .=> split(first(data.env), "_")[end],
                :strain .=> split(first(data.env), "_")[1],
                :latent1 => latent_rhvae[1, :],
                :latent2 => latent_rhvae[2, :],
                :drug => drug,
                :mse_full => logic50_full_mse_rhvae,
                :mse_crossval => logic50_crossval_mse_rhvae,
                :full_type => full_type,
                :crossval_type => crossval_type,
                :meta_idx => meta_idx,
                :model => "rhvae",
            )
        )

        # Map to VAE latent space if available
        if haskey(vae_dict, drug)
            # Run logic50_mean_std through VAE encoder
            latent_vae = vae_dict[drug].full.encoder(logic50_full_std).μ
            # Reconstruct logic50_mean_std
            logic50_full_recon_vae = vae_dict[drug].full.decoder(latent_vae).μ
            logic50_crossval_recon_vae = vae_dict[drug].crossval.decoder(latent_vae).μ
            # Compute reconstruction error
            logic50_full_mse_vae = Flux.mse(logic50_full_std, logic50_full_recon_vae)
            logic50_crossval_mse_vae = Flux.mse(logic50_crossval_std, logic50_crossval_recon_vae)

            # Append VAE latent coordinates to dataframe
            DF.append!(
                df_latent,
                DF.DataFrame(
                    :day .=> first(data.day),
                    :strain_num .=> first(data.strain_num),
                    :meta .=> first(data.env),
                    :env .=> split(first(data.env), "_")[end],
                    :strain .=> split(first(data.env), "_")[1],
                    :latent1 => latent_vae[1, :],
                    :latent2 => latent_vae[2, :],
                    :drug => drug,
                    :mse_full => logic50_full_mse_vae,
                    :mse_crossval => logic50_crossval_mse_vae,
                    :full_type => full_type,
                    :crossval_type => crossval_type,
                    :meta_idx => meta_idx,
                    :model => "vae",
                )
            )
        end
    end # for drug
end # for data

## =============================================================================

println("Plotting SVD cross validation results...")

# Define number of rows and columns
cols = length(drug_list) ÷ 2
rows = 2

# Initialize figure
fig = Figure(size=(700, 600))

# ------------------------------------------------------------------------------
# Plot layout
# ------------------------------------------------------------------------------

# Add global GridLayout
gl = GridLayout(fig[1, 1])

# Add grid layout for fig07A section banner
gl07A_banner = gl[1, 1] = GridLayout()
# Add grid layout for fig07A
gl07A = gl[2, 1] = GridLayout()

# Add grid layout for fig07B section banner
gl07B_banner = gl[3, 1] = GridLayout()
# Add grid layout for fig07B
gl07B = gl[4, 1] = GridLayout()

# ------------------------------------------------------------------------------
# Add section banners
# ------------------------------------------------------------------------------

# Add box for section title
Box(
    gl07A_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-25, right=-10) # Moves box to the left and right
)

# Add section title
Label(
    gl07A_banner[1, 1],
    "schematic of out-of-sample prediction training",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
    alignmode=Mixed(; left=-15) # Moves text to the left
)

# Add box for section title
Box(
    gl07B_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-25, right=-10) # Moves box to the left and right
)

# Add section title
Label(
    gl07B_banner[1, 1],
    "comparison of prediction accuracy for out-of-sample data",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
    alignmode=Mixed(; left=-15) # Moves text to the left
)

# ------------------------------------------------------------------------------
# Adjust subplot proportions
# ------------------------------------------------------------------------------

# Adjust row sizes
rowsize!(gl, 2, Auto(1))
rowsize!(gl, 4, Auto(2))

# ------------------------------------------------------------------------------
# Plot fig07B
# ------------------------------------------------------------------------------

# Loop through drugs
for (i, drug) in enumerate(drug_list)
    # Define row and column
    row = (i - 1) ÷ cols + 1
    col = mod1(i, cols)
    # Set axis
    ax = Axis(
        gl07B[row, col],
        title=drug,
        aspect=AxisAspect(4 / 3),
        yticks=WilkinsonTicks(3; k_min=3, k_max=4),
    )
    # Extract data
    data = df_svd[df_svd.drug.==drug, :]
    # Plot data
    scatterlines!(
        ax,
        data.rank,
        data.mse,
        label="SVD",
        color=Antibiotic.viz.colors()[:dark_gold],
    )

    # Extract rhvae mse for validation data
    mse_rhvae = df_latent[
        (df_latent.drug.==drug).&(df_latent.crossval_type.=="validation").&(df_latent.model.=="rhvae"),
        :mse_crossval
    ]

    # Extract vae mse for validation data if available
    if haskey(vae_dict, drug)
        mse_vae = df_latent[
            (df_latent.drug.==drug).&(df_latent.crossval_type.=="validation").&(df_latent.model.=="vae"),
            :mse_crossval
        ]

        # Plot vae mse as horizontal line
        hlines!(
            ax,
            StatsBase.mean(mse_vae),
            linestyle=:dash,
            label="2D VAE",
            linewidth=2,
            color=Antibiotic.viz.colors()[:dark_blue]
        )
    end

    # Plot rhvae mse as horizontal line
    hlines!(
        ax,
        StatsBase.mean(mse_rhvae),
        linestyle=:dash,
        label="2D RHVAE",
        linewidth=2,
        color=Antibiotic.viz.colors()[:dark_red]
    )

    if i == length(drug_list)
        # Add legend
        leg = Legend(
            gl07B[1, :, Top()],
            ax,
            orientation=:horizontal,
            merge=true,
            tellwidth=false,
            tellheight=false,
            framevisible=false,
            padding=(0, 0, 25, 0),
            labelsize=12,
        )
    end
end # for drug

# Add global axis labels
Label(
    gl07B[end, :, Bottom()],
    "number of SVD components",
    fontsize=14,
    padding=(0, 0, 0, 20),
)
Label(
    gl07B[:, 1, Left()],
    "mean squared error",
    fontsize=14,
    rotation=π / 2,
    padding=(-15, 40, 0, 0),
)

# Adjust gap between rows and columns
rowgap!(gl07B, -15)
colgap!(gl07B, 15)

# ------------------------------------------------------------------------------
# Add subplot labels
# ------------------------------------------------------------------------------

println("Adding subplot labels...")

Label(
    gl07A_banner[1, 1, Left()], "(A)",
    fontsize=20,
    padding=(0, 30, 0, 0),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

# Add subplot labels
Label(
    gl07B_banner[1, 1, Left()], "(B)",
    fontsize=20,
    padding=(0, 30, 0, 0),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

# Save figure
save("$(fig_dir)/fig07_v01.pdf", fig)

fig
