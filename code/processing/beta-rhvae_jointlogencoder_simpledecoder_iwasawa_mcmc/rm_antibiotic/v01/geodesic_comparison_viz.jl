## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic

# Import AutoEncoderToolkit to train VAEs
import AutoEncoderToolkit as AET

# Import libraries to handle data
import Glob
import CSV
import DataFrames as DF

# Import ML libraries
import Flux

# Import library to save models
import JLD2

# Import basic math
import StatsBase
import Random
Random.seed!(42)

# Load Plotting packages
using CairoMakie
using Makie
import ColorSchemes
import Colors
using PDFmerger: append_pdf!
# Activate backend
CairoMakie.activate!()

# Set plotting style
Antibiotic.viz.theme_makie!()

## =============================================================================

# Locate current directory
path_dir = pwd()
# Find the path prefix where to put figures
path_prefix = replace(
    match(r"processing/(.*)", path_dir).match,
    "processing" => "",
)

# Define data directory
data_dir = "$(git_root())/output/mcmc_iwasawa_logistic"
# Define output directory
out_dir = "$(git_root())/output$(path_prefix)"
# Define model state directory
rhvae_state_dir = "$(git_root())/output$(path_prefix)/rhvae_model_state"
# Define model crossvalidation directory
rhvae_crossval_dir = "$(git_root())/output$(path_prefix)/rhvae_crossvalidation_state"
# Define model state directory
vae_state_dir = "$(git_root())/output$(path_prefix)/vae_model_state"
# Define model crossvalidation directory
vae_crossval_dir = "$(git_root())/output$(path_prefix)/vae_crossvalidation_state"

# Define figure directory
fig_dir = "$(git_root())/fig$(path_prefix)"

# Create figure directory if it does not exist
if !isdir(fig_dir)
    println("Creating figure directory...")
    mkpath(fig_dir)
end

## =============================================================================

println("Loading data into memory...")

# Define data directory
data_dir = "$(git_root())/output/mcmc_iwasawa_logistic"

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

println("Loading RHVAE metadata...")

# Define loss function hyper-parameters
ϵ = Float32(1E-3) # Leapfrog step size
K = 10 # Number of leapfrog steps
βₒ = 0.3f0 # Initial temperature for tempering

# Define RHVAE hyper-parameters in a NamedTuple
rhvae_kwargs = (
    K=K,
    ϵ=ϵ,
    βₒ=βₒ,
)

# List model states
model_states = sort(
    Glob.glob("$(rhvae_state_dir)/beta-rhvae_*_epoch*.jld2"[2:end], "/")
)

# List crossvalidation states
crossval_states = sort(
    Glob.glob("$(rhvae_crossval_dir)/beta-rhvae_*_crossval*_epoch*.jld2"[2:end], "/")
)

# Initialize dataframe to store files metadata
df_meta = []

# Loop over model states
for file in model_states
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
    df_tmp = Dict(
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
    push!(df_meta, df_tmp)
end # for model_state

# Loop over crossvalidation states
for file in crossval_states
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
    df_tmp = Dict(
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
    push!(df_meta, df_tmp)
end # for model_state

## =============================================================================

println("Loading VAE metadata...")

# List model states
model_states = sort(
    Glob.glob("$(vae_state_dir)/beta-vae_*_epoch*.jld2"[2:end], "/")
)

# List crossvalidation states
crossval_states = sort(
    Glob.glob("$(vae_crossval_dir)/beta-vae_*_crossval*_epoch*.jld2"[2:end], "/")
)

# Loop over model states
for file in model_states
    # Extract epoch number from file name using regular expression
    epoch = parse(Int, match(r"epoch(\d+)", file).captures[1])
    # Extract drug name from file name using regular expression
    drug = match(r"beta-vae_(.+?)rm_epoch", split(file, "/")[end]).captures[1]
    # Load model_state file
    model_load = JLD2.load(file)
    # Extract loss train and loss val
    loss_train = model_load["loss_train"]
    loss_val = model_load["loss_val"]
    # Extract mse train and mse val
    mse_train = model_load["mse_train"]
    mse_val = model_load["mse_val"]
    # Generate temporary dataframe to store metadata
    df_tmp = Dict(
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
    push!(df_meta, df_tmp)
end # for model_state

# Loop over crossvalidation states
for file in crossval_states
    # Extract epoch number from file name using regular expression
    epoch = parse(Int, match(r"epoch(\d+)", file).captures[1])
    # Extract drug name from file name using regular expression
    drug = match(r"beta-vae_(.+?)rm_crossval", split(file, "/")[end]).captures[1]
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
    df_tmp = Dict(
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
    push!(df_meta, df_tmp)
end # for model_state

# Convert dictionary to dataframe
df_meta = DF.DataFrame(df_meta)

## =============================================================================

println("Plotting MSE...")

println("Plotting MSE...")

# Group by drug 
df_group = DF.groupby(df_meta, [:drug])

# Define number of rows and columns
cols = length(df_group)
rows = 2

# Initialize figure
fig = Figure(size=(210 * cols, 250 * rows))

# Add global GridLayout
gl = GridLayout(fig[1, 1])

# Set grid layout for original and crossvalidation models
gl_original = GridLayout(gl[1, 1])
gl_crossval = GridLayout(gl[2, 1])

# Set grid layout for legend
gl_legend = GridLayout(gl[1:2, 2])

# Define colors
colors = Dict(
    "rhvae" => ColorSchemes.Paired_10[1:2],
    "vae" => ColorSchemes.Paired_10[3:4],
)
# Loop over groups
for (i, df_drug) in enumerate(df_group)
    # Extract drug name
    drug = first(df_drug.drug)
    # Set axis for original model
    ax_full = Axis(
        gl_original[1, i],
        title=drug,
        xlabel="epoch",
        ylabel="MSE",
    )
    # Set axis for crossvalidation model
    ax_crossval = Axis(
        gl_crossval[1, i],
        title=drug,
        xlabel="epoch",
        ylabel="MSE",
    )

    # Group by model
    data_group = DF.groupby(df_drug, [:model])

    # Loop over models
    for (j, data_model) in enumerate(data_group)
        # Extract model type
        model = first(data_model.model)
        # Extract original model
        data_full = DF.sort(
            data_model[data_model.model_type.=="original", :],
            [:epoch],
        )
        # Extract crossvalidation model
        data_crossval = DF.sort(
            data_model[data_model.model_type.=="crossvalidation", :],
            [:epoch],
        )

        # Plot loss train and loss val
        lines!(
            ax_full,
            data_full.epoch,
            data_full.mse_train,
            label="$model train",
            color=colors[model][1],
        )
        lines!(
            ax_full,
            data_full.epoch,
            data_full.mse_val,
            label="$model validation",
            color=colors[model][2],
        )
        # Plot loss train and loss val
        lines!(
            ax_crossval,
            data_crossval.epoch,
            data_crossval.mse_train,
            label="$model train",
            color=colors[model][1],
        )
        lines!(
            ax_crossval,
            data_crossval.epoch,
            data_crossval.mse_val,
            label="$model validation",
            color=colors[model][2],
        )

    end # for model
    # Set legend
    Legend(
        gl_legend[1, 1],
        ax_full,
        merge=false,
    )
end # for drug

# Set global title for each grid layout
Label(
    gl[1, 1, Top()],
    "full model",
    fontsize=25,
    padding=(0, 0, 30, 0),
)
Label(
    gl[2, 1, Top()],
    "crossvalidation",
    fontsize=25,
    padding=(0, 0, 30, 0),
)

# Separate rows
rowgap!(gl, 25)

# Save figure
save("$(fig_dir)/vae_rhvae_mse.pdf", fig)

fig