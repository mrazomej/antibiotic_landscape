## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic

# Import package for VAEs
import AutoEncode
import AutoEncode.diffgeo.NeuralGeodesics as NG

# Import library to handle data
import Glob
import CSV
import DataFrames as DF

# Import ML libraries
import Flux

# Import library to save models
import JLD2

# Import Plotting libraries
using CairoMakie
import ColorSchemes
import PDFmerger

# Activate backend
CairoMakie.activate!()

# Set Plotting style
Antibiotic.viz.theme_makie!()

# Import basic math
import LinearAlgebra
import Random
import StatsBase

Random.seed!(42)

## =============================================================================

# Define model temperature
T = 0.5f0
# Define number of time points to evaluate along curve
n_time = 100

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

println("Loading NeuralGeodesic model...")

# Define model directory
model_dir = "$(git_root())/code/processing/" *
            "beta-rhvae_jointlogencoder_simpledecoder_iwasawa_fitness/" *
            "v02/output"

# Load model template
nng_template = JLD2.load("$(model_dir)/geodesic.jld2")["model"]

# List all parameter files
nng_files = Glob.glob("$(model_dir)/geodesic_state/*$(T)temp*.jld2"[2:end], "/")

## =============================================================================

println("Loading RHVAE models...")

# List files in output directory
files = Glob.glob("$(model_dir)/model_state/*$(T)temp*.jld2"[2:end], "/")

# Load model template
rhvae = JLD2.load("$(model_dir)/model.jld2")["model"]

# Modify temperature
rhvae = AutoEncode.RHVAEs.RHVAE(
    deepcopy(rhvae.vae),
    deepcopy(rhvae.metric_chain),
    deepcopy(rhvae.centroids_data),
    deepcopy(rhvae.centroids_latent),
    deepcopy(rhvae.L),
    deepcopy(rhvae.M),
    T,
    deepcopy(rhvae.λ)
)

# Search for model files
model_files = sort(Glob.glob(
    "$(model_dir)/model_state/*$(T)*.jld2"[2:end], "/"
))
# Load parameters
model_state = JLD2.load(model_files[end])["model_state"]
# Set model parameters
Flux.loadmodel!(rhvae, model_state)
# Update metric
AutoEncode.RHVAEs.update_metric!(rhvae)

## =============================================================================

println("Compute latent space metric")

# Define number of points per axis
n_points = 300

# Define range of latent space
latent_range_z1 = Float32.(range(-3.2, 3.2, length=n_points))
latent_range_z2 = Float32.(range(-3.5, 3, length=n_points))

# Define latent points to evaluate
z_mat = reduce(hcat, [[x, y] for x in latent_range_z1, y in latent_range_z2])

# Compute inverse metric tensor
Ginv = AutoEncode.RHVAEs.G_inv(z_mat, rhvae)

# Compute log determinant of metric tensor
logdetG = reshape(-1 / 2 * AutoEncode.utils.slogdet(Ginv), n_points, n_points);

## =============================================================================

println("Compute PCA")

# Perform SVD on the data
ic50_svd = LinearAlgebra.svd(ic50_std)

# Extract principal components
pcs = ic50_svd.U

# Project data to the first two principal components
data_pca = pcs[:, 1:2]' * ic50_std

# Convert data to DataFrame
df_pca = DF.DataFrame(
    data_pca',
    [:pc1, :pc2],
)

# Invert value of first principal component
df_pca.pc1 = -df_pca.pc1

# Extract strains as ordered in ic50 matrix
strains_mat = [x.strain for x in keys(df_group)]
day_mat = [x.day for x in keys(df_group)]

# Add strains and days to DataFrame
DF.insertcols!(
    df_pca,
    :strain => strains_mat,
    :day => day_mat
)

# Add corresponding metadata resistance value
df_pca = DF.leftjoin!(
    df_pca,
    unique(df_ic50[:, [:strain, :day, :parent, :env]]),
    on=[:strain, :day]
)

## =============================================================================

println("Encode data to RHVAE latent space")

# Project data to RHVAE latent space
data_latent = rhvae.vae.encoder(ic50_std).µ

# Convert data to DataFrame
df_latent = DF.DataFrame(
    data_latent',
    [:z1, :z2],
)

# Extract strains as ordered in ic50 matrix
strains_mat = [x.strain for x in keys(df_group)]
day_mat = [x.day for x in keys(df_group)]

# Add strains and days to DataFrame
DF.insertcols!(
    df_latent,
    :strain => strains_mat,
    :day => day_mat
)

# Add corresponding metadata resistance value
df_latent = DF.leftjoin!(
    df_latent,
    unique(df_ic50[:, [:strain, :day, :parent, :env]]),
    on=[:strain, :day]
)

## =============================================================================

println("Merge PCA and latent data")

# Merge PCA and latent data
df_latent = DF.leftjoin(df_pca, df_latent, on=[:strain, :day, :parent, :env])

## =============================================================================

println("Setting output directories...")

# Define output directory
out_dir = "./output/fig"

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

## =============================================================================

# Define unique environments
envs = unique(df_latent.env)

## =============================================================================

# Loop through environments
for (k, env) in enumerate(envs)
    # Group environment data by parent
    df_parent_group = DF.groupby(df_latent[(df_latent.env.==env), :], :parent)

    # Define file name
    fname = "$(out_dir)/geodesics_replicate_PCA_vs_RHVAE_$(env)env.pdf"

    # Loop through parents
    for (i, df_parent) in enumerate(df_parent_group)
        println("Processing $(env)env $(i)/$(length(df_parent_group))")
        # Initialize figure
        fig = Figure(size=(650, 300))
        # Add Grid layout
        gl = fig[1, 1] = GridLayout()
        # Add axis
        axes = [
            Axis(
                gl[1, 1],
                xlabel="PC1",
                ylabel="PC2",
                aspect=AxisAspect(1),
            ),
            Axis(
                gl[1, 2],
                xlabel="latent dimension 1",
                ylabel="latent dimension 2",
                aspect=AxisAspect(1),
            ),
        ]

        # Plot PCA points as gray background
        scatter!(
            axes[1],
            df_latent.pc1,
            df_latent.pc2,
            markersize=5,
            color=(:gray, 0.5)
        )

        # Plot heatmap of log determinant of metric tensor
        hm = heatmap!(
            axes[2], latent_range_z1, latent_range_z2, logdetG, colormap=:tokyo
        )

        # Add colorbar
        cb = Colorbar(
            gl[1, 3],
            hm,
            size=8,
            label="log√det(G)",
            labelsize=12,
            labelpadding=0.0,
            ticklabelsize=12,
            ticksvisible=false
        )
        # Group data by strain
        df_strain_group = DF.groupby(df_parent, :strain)
        # Loop through strains
        for (j, df_strain) in enumerate(df_strain_group)
            # Sort data by day
            DF.sort!(df_strain, :day)

            # %%%%%%%%%%%%%%%%%%% latent trajectories %%%%%%%%%%%%%%%%%%%

            # Plot PCA trajectory
            scatterlines!(
                axes[1],
                df_strain.pc1,
                df_strain.pc2,
                color=ColorSchemes.Paired_12[2:2:end][j],
                linewidth=2,
            )
            # Plot RHVAE trajectory
            scatterlines!(
                axes[2],
                df_strain.z1,
                df_strain.z2,
                color=ColorSchemes.Paired_12[2:2:end][j],
                linewidth=2,
            )
        end # for df_strain

        # %%%%%%%%%%%%%%%%%%% geodesic trajectories %%%%%%%%%%%%%%%%%%%
        for (j, df_strain) in enumerate(df_strain_group)
            # Extract strain
            strain = first(df_strain.strain)
            # Plot PCA geodesic
            lines!(
                axes[1],
                [df_strain[1, :pc1], df_strain[end, :pc1]],
                [df_strain[1, :pc2], df_strain[end, :pc2]],
                linewidth=3,
                linestyle=Linestyle([0.5, 1.0, 1.5, 2.5]),
                color=ColorSchemes.Paired_12[1:2:end][j],
            )

            # Plot RHVAE geodesic

            # Search for model files
            file = Glob.glob(
                "$(model_dir)/geodesic_state/"[2:end] *
                "neuralgeodesic_$(T)temp_$(env)env_" *
                "$(replace(strain, " " => "-"))strain*", "/"
            )[1]

            # Load file
            nng_dict = JLD2.load(file)
            # Define NeuralGeodesic model
            nng = NG.NeuralGeodesic(
                nng_template.mlp,
                nng_dict["z_init"],
                nng_dict["z_end"],
            )
            # Load parameters
            Flux.loadmodel!(nng, nng_dict["model_state"])

            # Generate curve
            curve = nng(t_array)

            # Add geodesic to axis
            lines!(
                axes[2],
                curve[1, :],
                curve[2, :],
                linewidth=3,
                linestyle=Linestyle([0.5, 1.0, 1.5, 2.5]),
                color=ColorSchemes.Paired_12[1:2:end][j],
            )

        end # for

        # %%%%%%%%%%%%%%%%%%% first and last point %%%%%%%%%%%%%%%%%%%
        for (j, df_strain) in enumerate(df_strain_group)
            # Add first point PCA
            scatter!(
                axes[1],
                [df_strain.pc1[1]],
                [df_strain.pc2[1]],
                color=:white,
                markersize=18,
                marker=:xcross
            )
            scatter!(
                axes[1],
                [df_strain.pc1[1]],
                [df_strain.pc2[1]],
                color=ColorSchemes.Paired_12[1:2:end][j],
                markersize=10,
                marker=:xcross
            )

            # Add last point PCA
            scatter!(
                axes[1],
                [df_strain.pc1[end]],
                [df_strain.pc2[end]],
                color=:white,
                markersize=18,
                marker=:utriangle
            )
            scatter!(
                axes[1],
                [df_strain.pc1[end]],
                [df_strain.pc2[end]],
                color=ColorSchemes.Paired_12[1:2:end][j],
                markersize=10,
                marker=:utriangle
            )

            # Add first point RHVAE
            scatter!(
                axes[2],
                [df_strain.z1[1]],
                [df_strain.z2[1]],
                color=:white,
                markersize=18,
                marker=:xcross
            )
            scatter!(
                axes[2],
                [df_strain.z1[1]],
                [df_strain.z2[1]],
                color=ColorSchemes.Paired_12[1:2:end][j],
                markersize=10,
                marker=:xcross
            )

            # Add last point RHVAE
            scatter!(
                axes[2],
                [df_strain.z1[end]],
                [df_strain.z2[end]],
                color=:white,
                markersize=18,
                marker=:utriangle
            )
            scatter!(
                axes[2],
                [df_strain.z1[end]],
                [df_strain.z2[end]],
                color=ColorSchemes.Paired_12[1:2:end][j],
                markersize=10,
                marker=:utriangle
            )

        end # for df_strain


        # %%%%%%%%%%%%%%%%%%% add labels %%%%%%%%%%%%%%%%%%%
        # Add subplots subtitle
        Label(
            gl[1, :, Top()],
            "$(env) | $(first(df_parent.parent))",
            valign=:bottom,
            font=:bold,
            padding=(0, 0, 5, 0)
        )


        # Save figure into temporary PDF
        save("$(out_dir)/temp.pdf", fig)

        # Merge PDFs
        PDFmerger.append_pdf!(
            fname, "$(out_dir)/temp.pdf", cleanup=true
        )
    end # for df_parent
end # for env