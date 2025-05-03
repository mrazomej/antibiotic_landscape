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

println("Loading errors...")

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
df_meta = DF.DataFrame()

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
    df_tmp = DF.DataFrame(
        :epoch => epoch,
        :drug => drug,
        :loss_train => loss_train,
        :loss_val => loss_val,
        :mse_train => mse_train,
        :mse_val => mse_val,
        :model_file => file,
        :model_state => file,
        :model_type => "original",
        :train_frac => 0.85,
    )
    # Append temporary dataframe to main dataframe
    global df_meta = DF.vcat(df_meta, df_tmp)
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
    df_tmp = DF.DataFrame(
        :epoch => epoch,
        :drug => drug,
        :loss_train => loss_train,
        :loss_val => loss_val,
        :mse_train => mse_train,
        :mse_val => mse_val,
        :model_file => file,
        :model_state => file,
        :model_type => "crossvalidation",
        :train_frac => split_frac,
    )
    # Append temporary dataframe to main dataframe
    global df_meta = DF.vcat(df_meta, df_tmp)
end # for model_state

## =============================================================================

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

# Loop over groups
for (i, data) in enumerate(df_group)
    # Extract drug name
    drug = data.drug[1]
    # Extract original model
    data_full = data[data.model_type.=="original", :]
    # Extract crossvalidation model
    data_crossval = data[data.model_type.=="crossvalidation", :]
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

    # Plot loss train and loss val
    lines!(
        ax_full,
        data_full.epoch,
        data_full.mse_train,
        label="train",
    )
    lines!(
        ax_full,
        data_full.epoch,
        data_full.mse_val,
        label="validation",
    )
    # Plot loss train and loss val
    lines!(
        ax_crossval,
        data_crossval.epoch,
        data_crossval.mse_train,
        label="train",
    )
    lines!(
        ax_crossval,
        data_crossval.epoch,
        data_crossval.mse_val,
        label="validation",
    )

    # Set legend
    Legend(
        gl_legend[1, 1],
        ax_full,
        merge=true,
    )
end

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
save("$(fig_dir)/beta-rhvae_mse.png", fig)

fig

## =============================================================================

println("Loading models...")

# Group data by drug
df_group = DF.groupby(df_meta, [:drug])

# Initialize dictionary to store models
rhvae_dict = Dict{String,Any}()

# Loop over drugs
for (i, data) in enumerate(df_group)
    # Extract drug name
    drug = data.drug[1]
    # Extract original model
    data_full = DF.sort(data[data.model_type.=="original", :], [:epoch])
    # Extract crossvalidation model
    data_crossval = DF.sort(
        data[data.model_type.=="crossvalidation", :], [:epoch]
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

        # Run logic50_mean_std through encoder
        latent_full = rhvae_dict[drug].full(logic50_full_std; latent=true)
        # Extract latent coordinates
        latent = latent_full.phase_space.z_final
        # Reconstruct logic50_mean_std
        logic50_full_recon = rhvae_dict[drug].full.vae.decoder(latent).µ
        logic50_crossval_recon = rhvae_dict[drug].crossval.vae.decoder(latent).µ
        # Compute reconstruction error
        logic50_full_mse = Flux.mse(logic50_full_std, logic50_full_recon)
        logic50_crossval_mse = Flux.mse(logic50_crossval_std, logic50_crossval_recon)
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
                :drug => drug,
                :mse_full => logic50_full_mse,
                :mse_crossval => logic50_crossval_mse,
                :full_type => full_type,
                :crossval_type => crossval_type,
                :meta_idx => meta_idx,
            )
        )
    end # for drug
end # for data

## =============================================================================

println("Evaluate metric tensor for all drugs...")

# Initialize dictionary to store metric tensors
metric_dict = Dict{String,Any}()

# Define number of points
n_points = 100

# Loop over drugs
for (i, drug) in enumerate(drug_list)
    # Extract data
    data_drug = df_latent[df_latent.drug.==drug, :]
    # Defie latent space ranges
    latent1_range = range(
        minimum(data_drug.latent1) - 2.5,
        maximum(data_drug.latent1) + 2.5,
        length=n_points,
    )
    latent2_range = range(
        minimum(data_drug.latent2) - 2.5,
        maximum(data_drug.latent2) + 2.5,
        length=n_points,
    )
    # Define latent points to evaluate
    z_mat = reduce(hcat, [[x, y] for x in latent1_range, y in latent2_range])
    # Compute inverse metric tensor
    Ginv = AET.RHVAEs.G_inv(z_mat, rhvae_dict[drug].full)

    # Compute metric 
    logdetG = reshape(
        -1 / 2 * AET.utils.slogdet(Ginv), n_points, n_points
    )
    # Store in dictionary
    metric_dict[drug] = (
        latent1_range=latent1_range,
        latent2_range=latent2_range,
        logdetG=logdetG,
    )
end # for drug

## =============================================================================

println("Plotting latent space for training data...")

# Define number of rows and columns
cols = length(drug_list) ÷ 2
rows = 2

# Initialize figure
fig = Figure(size=(200 * cols, 200 * rows + 100))

# Add global GridLayout
gl = GridLayout(fig[1, 1])

# Set grid layout for subplots
gl_plots = GridLayout(gl[1, 1])

# Set grid layout for legend
gl_legend = GridLayout(gl[2, 1])

# Loop over drugs
for (i, drug) in enumerate(drug_list)
    # Define row and column
    row = (i - 1) ÷ cols + 1
    col = (i - 1) % cols + 1
    # Set axis
    ax = Axis(
        gl_plots[row, col],
        title=drug,
        aspect=AxisAspect(1),
    )
    hidedecorations!(ax)

    # Plot heatmpat of log determinant of metric tensor
    hm = heatmap!(
        ax,
        metric_dict[drug].latent1_range,
        metric_dict[drug].latent2_range,
        metric_dict[drug].logdetG,
        colormap=Reverse(to_colormap(ColorSchemes.PuBu)),
    )
    # Extract data
    data_train = df_latent[(df_latent.drug.==drug).&(df_latent.full_type.=="train"), :]
    data_val = df_latent[(df_latent.drug.==drug).&(df_latent.full_type.=="validation"), :]
    # Define fraction of train data
    frac_train = length(data_train.latent1) / (
        length(data_train.latent1) + length(data_val.latent1))
    # Plot data
    scatter!(
        ax,
        data_train.latent1,
        data_train.latent2,
        color=(:white, 0.5),
        markersize=6,
        label="train ($(round(frac_train * 100))% of data)",
    )
    scatter!(
        ax,
        data_val.latent1,
        data_val.latent2,
        color=(Antibiotic.viz.colors()[:dark_red], 1),
        markersize=6,
        label="validation ($(round((1 - frac_train) * 100))% of data)",
    )

    # Add legend
    leg = Legend(
        gl_legend[1, 1],
        ax,
        "7 drugs train",
        orientation=:horizontal,
        merge=true,
    )
end # for drug

# Save figure
save("$(fig_dir)/beta-rhvae_latent_space.png", fig)

fig

## =============================================================================

println("Plotting MSE distribution...")

Random.seed!(42)

# Group data by drug
df_group = DF.groupby(df_latent, [:drug])
# Extract drug list
drug_list = [first(d.drug) for d in df_group]

# Initialize figure
fig = Figure(size=(600, 600))

# Add axes
ax1 = Axis(
    fig[1, 1],
    xlabel="excluded drug",
    ylabel="MSE on logIC₅₀\nreconstruction",
    xticks=(1:length(df_group), drug_list),
    title="(left = train 7 drugs, right = validate 1 drug)"
)

ax2 = Axis(
    fig[2, 1],
    xlabel="excluded drug",
    ylabel="MSE on logIC₅₀\nreconstruction",
    xticks=(1:length(df_group), drug_list),
    yscale=log10,
    title="(left = train 7 drugs, right = validate 1 drug)"
)

# Define colors
colors = ColorSchemes.Paired_12

# Define displacement
disp = 0.15

# Loop through groups
for (i, data) in enumerate(df_group)
    # Calculate color indices for this pair, wrapping around if needed
    color_idx = mod1.([(2 * i - 1), 2 * i], length(colors))

    # Plot on linear scale axis
    scatter!(
        ax1,
        i .- disp .+ randn(length(data.mse_full)) .* 0.02,
        data.mse_full,
        label="full",
        color=(colors[color_idx[1]], 0.15),
    )
    boxplot!(
        ax1,
        repeat([i - disp], length(data.mse_full)),
        data.mse_full,
        color=(colors[color_idx[1]], 1.0),
        width=0.25,
        show_outliers=false,
    )
    scatter!(
        ax1,
        i .+ disp .+ randn(length(data.mse_crossval)) .* 0.02,
        data.mse_crossval,
        label="crossvalidation",
        color=(colors[color_idx[2]], 0.15),
    )
    boxplot!(
        ax1,
        repeat([i + disp], length(data.mse_crossval)),
        data.mse_crossval,
        color=(colors[color_idx[2]], 1.0),
        width=0.25,
        show_outliers=false,
    )

    # Plot on log scale axis
    scatter!(
        ax2,
        i .- disp .+ randn(length(data.mse_full)) .* 0.02,
        data.mse_full,
        label="full",
        color=(colors[color_idx[1]], 0.15),
    )
    boxplot!(
        ax2,
        repeat([i - disp], length(data.mse_full)),
        data.mse_full,
        color=(colors[color_idx[1]], 1.0),
        width=0.25,
        show_outliers=false,
    )
    scatter!(
        ax2,
        i .+ disp .+ randn(length(data.mse_crossval)) .* 0.02,
        data.mse_crossval,
        label="crossvalidation",
        color=(colors[color_idx[2]], 0.15),
    )
    boxplot!(
        ax2,
        repeat([i + disp], length(data.mse_crossval)),
        data.mse_crossval,
        color=(colors[color_idx[2]], 1.0),
        width=0.25,
        show_outliers=false,
    )
end

# Save figure
save("$(fig_dir)/beta-rhvae_mse_distribution.png", fig)

fig

## =============================================================================

println("Plotting MSE distribution...")

Random.seed!(42)

# Group data by drug
df_group = DF.groupby(df_latent, [:drug])
# Extract drug list
drug_list = [first(d.drug) for d in df_group]

# Initialize figure
fig = Figure(size=(600, 600))

# Add global GridLayout
gl = GridLayout(fig[1, 1])

# Add axes
ax1 = Axis(
    gl[1, 1],
    xlabel="excluded drug",
    ylabel="MSE on logIC₅₀\nreconstruction",
    xticks=(1:length(df_group), drug_list),
)

ax2 = Axis(
    gl[2, 1],
    xlabel="excluded drug",
    ylabel="MSE on logIC₅₀\nreconstruction",
    xticks=(1:length(df_group), drug_list),
    yscale=log10,
)

# Define colors
colors = ColorSchemes.Paired_12

# Define displacement
disp = 0.15

# Loop through groups
for (i, data) in enumerate(df_group)
    # Calculate color indices for this pair, wrapping around if needed
    color_idx = mod1.([(2 * i - 1), 2 * i], length(colors))

    # Extract train data
    data_train = data[data.crossval_type.=="train", :]
    # Extract validation data
    data_val = data[data.crossval_type.=="validation", :]
    # Define fraction of train data
    frac_train = length(data_train.latent1) / (
        length(data_train.latent1) + length(data_val.latent1))
    # Plot on linear scale axis
    scatter!(
        ax1,
        i .- disp .+ randn(length(data_train.mse_crossval)) .* 0.02,
        data_train.mse_crossval,
        label="train",
        color=(colors[color_idx[1]], 0.15),
    )
    boxplot!(
        ax1,
        repeat([i - disp], length(data_train.mse_crossval)),
        data_train.mse_crossval,
        color=(colors[color_idx[1]], 1.0),
        width=0.25,
        show_outliers=false,
    )
    scatter!(
        ax1,
        i .+ disp .+ randn(length(data_val.mse_crossval)) .* 0.02,
        data_val.mse_crossval,
        label="validation ($(round((1 - frac_train) * 100))% of data)",
        color=(colors[color_idx[2]], 0.15),
    )
    boxplot!(
        ax1,
        repeat([i + disp], length(data_val.mse_crossval)),
        data_val.mse_crossval,
        color=(colors[color_idx[2]], 1.0),
        width=0.25,
        show_outliers=false,
    )

    # Plot on log scale axis
    scatter!(
        ax2,
        i .- disp .+ randn(length(data_train.mse_crossval)) .* 0.02,
        data_train.mse_crossval,
        label="train",
        color=(colors[color_idx[1]], 0.15),
    )
    boxplot!(
        ax2,
        repeat([i - disp], length(data_train.mse_crossval)),
        data_train.mse_crossval,
        color=(colors[color_idx[1]], 1.0),
        width=0.25,
        show_outliers=false,
    )
    scatter!(
        ax2,
        i .+ disp .+ randn(length(data_val.mse_crossval)) .* 0.02,
        data_val.mse_crossval,
        label="validation",
        color=(colors[color_idx[2]], 0.15),
    )
    boxplot!(
        ax2,
        repeat([i + disp], length(data_val.mse_crossval)),
        data_val.mse_crossval,
        color=(colors[color_idx[2]], 1.0),
        width=0.25,
        show_outliers=false,
    )
    # Set global title
    Label(
        gl[1, 1, Top()],
        "left = train ($(round(frac_train * 100))% of data)\n" *
        "right = validation ($(round((1 - frac_train) * 100))% of data)",
        fontsize=18,
        padding=(0, 0, 10, 0),
    )
end

# Save figure
save("$(fig_dir)/beta-rhvae_mse_distribution_1drug.png", fig)

fig

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
    logic50_svd = logic50_mean[env_idx, vcat(val_idx, train_idx)]
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

println("Plotting SVD cross validation results...")

# Define number of rows and columns
cols = length(drug_list) ÷ 2
rows = 2

# Initialize figure
fig = Figure(size=(200 * cols, 200 * rows + 75))

# Add global GridLayout
gl = GridLayout(fig[1, 1])

# Add grid layout for plots
gl_plots = GridLayout(gl[1, 1])

# Add grid layout for legend
gl_legend = GridLayout(gl[2, 1])

# Loop through drugs
for (i, drug) in enumerate(drug_list)
    # Define row and column
    row = (i - 1) ÷ cols + 1
    col = mod1(i, cols)
    # Set axis
    ax = Axis(
        gl_plots[row, col],
        xlabel="SVD rank",
        ylabel="MSE",
        title=drug,
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
    # Extract rhvae mse
    mse_rhvae = df_latent[
        (df_latent.drug.==drug).&(df_latent.crossval_type.=="validation"),
        :mse_crossval
    ]
    # Plot rhvae mse as horizontal line
    hlines!(
        ax,
        StatsBase.mean(mse_rhvae),
        linestyle=:dash,
        label="2D RHVAE",
        linewidth=2,
        color=Antibiotic.viz.colors()[:dark_red]
    )

    # Add legend
    leg = Legend(
        gl_legend[1, 1],
        ax,
        orientation=:horizontal,
        merge=true,
        tellwidth=false,
    )

    # Add global title
    Label(
        gl[1, 1, Top()],
        "SVD vs. 2D RHVAE cross validation prediction",
        fontsize=22,
        padding=(0, 0, 30, 0),
    )
end # for drug

# Save figure
save("$(fig_dir)/beta-rhvae_svd_crossval.png", fig)

fig

## =============================================================================

println("Plotting MSE ECDF for SVD and 2D RHVAE...")

# Define number of rows and columns
cols = length(drug_list) ÷ 2
rows = 2

# Initialize figure
fig = Figure(size=(200 * cols, 200 * rows + 75))

# Add global GridLayout
gl = GridLayout(fig[1, 1])

# Add grid layout for plots
gl_plots = GridLayout(gl[1, 1])

# Add grid layout for legend
gl_legend = GridLayout(gl[2, 1])

# Loop through drugs
for (i, drug) in enumerate(drug_list)
    # Define row and column
    row = (i - 1) ÷ cols + 1
    col = mod1(i, cols)
    # Set axis
    ax = Axis(
        gl_plots[row, col],
        xlabel="MSE",
        ylabel="ECDF",
        title=drug,
    )
    # Extract validation idx
    val_idx = rhvae_dict[drug].crossval_val_idx
    # Extract validation data
    logic50_val = logic50_mean[i, val_idx]
    # Compute MSE for SVD
    mse_svd_rank2 = (vec(svd_pred[drug][2]) .- vec(logic50_val)) .^ 2
    mse_svd_rank7 = (vec(svd_pred[drug][7]) .- vec(logic50_val)) .^ 2
    # Compute ECDF for SVD rank 2 and 7
    ecdfplot!(
        ax,
        mse_svd_rank2,
        label="SVD (rank 2)",
        color=Antibiotic.viz.colors()[:dark_blue],
    )
    ecdfplot!(
        ax,
        mse_svd_rank7,
        label="SVD (rank 7)",
        color=Antibiotic.viz.colors()[:dark_gold],
    )
    # Extract rhvae mse
    mse_rhvae = df_latent[
        (df_latent.drug.==drug).&(df_latent.crossval_type.=="validation"),
        :mse_crossval
    ]
    # Compute ECDF
    ecdfplot!(
        ax,
        mse_rhvae,
        label="2D RHVAE",
        color=Antibiotic.viz.colors()[:dark_red],
    )

    # Add legend
    leg = Legend(
        gl_legend[1, 1],
        ax,
        orientation=:horizontal,
        merge=true,
    )
end # for drug

# Add global title
Label(
    gl[1, 1, Top()],
    "MSE distribution for SVD and 2D RHVAE cross validation predictions",
    fontsize=18,
    padding=(0, 0, 30, 0),
)

# Save figure
save("$(fig_dir)/beta-rhvae_svd_ecdf.png", fig)


fig

## =============================================================================