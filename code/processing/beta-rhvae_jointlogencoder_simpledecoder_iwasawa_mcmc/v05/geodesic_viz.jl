## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic

# Import AutoEncoderToolkit to train VAEs
import AutoEncoderToolkit as AET
import AutoEncoderToolkit.diffgeo.NeuralGeodesics as NG

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

# Define figure directory
fig_dir = "$(git_root())/fig$(out_prefix)"

# Create figure directory if it does not exist
if !isdir(fig_dir)
    println("Creating figure directory...")
    mkpath(fig_dir)
end

## =============================================================================

# List all files in the directory
geodesic_files = Glob.glob("$(geodesic_dir)/*.jld2"[2:end], "/")

# Initialize dataframe to store files metadata
df_meta = DF.DataFrame()

# Loop over geodesic state files
for gf in geodesic_files
    # Extract initial generation number from file name using regular expression
    day_init = parse(Int, match(r"dayinit(\d+)", gf).captures[1])
    # Extract final generation number from file name using regular expression
    day_final = parse(Int, match(r"dayfinal(\d+)", gf).captures[1])
    # Extract evo stress number from file name using regular expression
    env = match(r"evoenv(\w+)_id", gf).captures[1]
    # Extract GRN id from file name using regular expression
    strain_num = parse(Int, match(r"id(\d+)", gf).captures[1])
    # Extract RHVAE epoch number from file name using regular expression
    rhvae_epoch = parse(Int, match(r"rhvaeepoch(\d+)", gf).captures[1])
    # Extract geodesic epoch number from file name using regular expression
    geo_epoch = parse(Int, match(r"geoepoch(\d+)", gf).captures[1])
    # Append as DataFrame
    DF.append!(
        df_meta,
        DF.DataFrame(
            :day_init => day_init,
            :day_final => day_final,
            :env => env,
            :strain_num => strain_num,
            :rhvae_epoch => rhvae_epoch,
            :geodesic_epoch => geo_epoch,
            :geodesic_state => gf,
        ),
    )
end # for gf in geodesic_files

# Sort dataframe by environment
DF.sort!(df_meta, :env)

## =============================================================================

println("Loading NeuralGeodesic template...")
nng_template = JLD2.load("$(out_dir)/geodesic.jld2")["model"].mlp

# Define number of points per axis
n_time = 75
# Define time points along curve
t_array = Float32.(collect(range(0, 1, length=n_time)))

## =============================================================================

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

# Load logic50 data
df_logic50 = CSV.read("$(data_dir)/logic50_ci.csv", DF.DataFrame)
# Load list of drugs
drug_list = sort(unique(df_logic50.drug))

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
        )
    )
end # for 

## =============================================================================

println("Compute Riemannian metric for latent space...")

# Define number of points per axis
n_points = 100

# Extract latent space ranges
latent1_range = range(
    minimum(df_latent.latent1) - 1.5,
    maximum(df_latent.latent1) + 1.5,
    length=n_points
)
latent2_range = range(
    minimum(df_latent.latent2) - 1.5,
    maximum(df_latent.latent2) + 1.5,
    length=n_points
)
# Define latent points to evaluate
z_mat = reduce(hcat, [[x, y] for x in latent1_range, y in latent2_range])

# Compute inverse metric tensor
Ginv = AET.RHVAEs.G_inv(z_mat, rhvae)

# Compute metric 
logdetG = reshape(
    -1 / 2 * AET.utils.slogdet(Ginv), n_points, n_points
)


## =============================================================================

println("Plotting latent space geodesic trajectories...")

# Group data by :env
df_group = DF.groupby(df_latent, :env)

# Define figure name
fname = "$(fig_dir)/geodesic_latent_trajectory"

# If PDF exists, delete it
if isfile("$(fname).pdf")
    rm("$(fname).pdf")
end

# Loop through meta chunks
for (i, data_group) in enumerate(df_group)
    # Extract unique strain numbers
    strain_nums = unique(data_group.strain_num)
    # Define number of columns
    cols = 4
    # Define the number of needed rows
    rows = ceil(Int, length(strain_nums) / cols)

    # Initialize figure
    fig = Figure(size=(200 * cols, 200 * rows))
    # Add grid layout
    gl = fig[1, 1] = GridLayout()
    # Loop through metadata
    for (j, strain_num) in enumerate(strain_nums)
        # Extract metadata
        data_meta = df_meta[
            (df_meta.env.==first(data_group.env)).&(df_meta.strain_num.==strain_num),
            :]

        println("   - Plotting geodesic: $(strain_num)")
        # Define row and column index
        row = (j - 1) ÷ cols + 1
        col = (j - 1) % cols + 1
        # Add axis
        ax = Axis(
            gl[row, col],
            aspect=AxisAspect(1),
            title="env: $(first(data_meta.env)) | " *
                  "strain: $(first(data_meta.strain_num))",
            xticksvisible=false,
            yticksvisible=false,
        )
        # Hide axis labels
        hidedecorations!(ax)

        # Plot heatmap of log determinant of metric tensor
        heatmap!(
            ax,
            latent1_range,
            latent2_range,
            logdetG,
            colormap=Reverse(to_colormap(ColorSchemes.PuBu)),
            alpha=0.5,
        )

        # Extract lineage information
        lineage = df_latent[df_latent.strain_num.==data_meta.strain_num, :]

        # Plot lineage
        scatterlines!(
            ax,
            lineage.latent1,
            lineage.latent2,
            color=Antibiotic.viz.colors()[:gold],
            markersize=6,
            linewidth=2,
        )

        # Load geodesic state
        geo_state = JLD2.load(first(data_meta.geodesic_state))
        # Define NeuralGeodesic model
        nng = NG.NeuralGeodesic(
            nng_template,
            geo_state["latent_init"],
            geo_state["latent_end"],
        )
        # Update model state
        Flux.loadmodel!(nng, geo_state["model_state"])
        # Generate curve
        curve = nng(t_array)
        # Add geodesic line to axis
        lines!(
            ax,
            eachrow(curve)...,
            color=Antibiotic.viz.colors()[:dark_red],
            linewidth=2.5,
        )

        # Add first point 
        scatter!(
            ax,
            [lineage.latent1[1]],
            [lineage.latent2[1]],
            color=Antibiotic.viz.colors()[:dark_red],
            markersize=12,
            marker=:xcross
        )

        # Add last point
        scatter!(
            ax,
            [lineage.latent1[end]],
            [lineage.latent2[end]],
            color=Antibiotic.viz.colors()[:dark_red],
            markersize=12,
            marker=:utriangle
        )
    end # for data_meta in eachrow(chunk)
    # Save figure as PDF
    save("tmp.pdf", fig)
    # Append to final figure
    append_pdf!("$(fname).pdf", "tmp.pdf", cleanup=true)
    # Save figure as PNG
    save("$(fname)_$(first(data_group.env)).png", fig)
end # for chunk in df_meta_chunks

## =============================================================================

# Group data by :env
df_group = DF.groupby(df_latent, :env)

# Define figure name
fname = "$(fig_dir)/geodesic_latent_trajectory_brownian"

# If PDF exists, delete it
if isfile("$(fname).pdf")
    rm("$(fname).pdf")
end

# Define number of Brownian bridges
n_bridges = 100
# Define sigma
sigma = 2.0

# Loop through meta chunks
for (i, data_group) in enumerate(df_group)
    # Extract unique strain numbers
    strain_nums = unique(data_group.strain_num)
    # Define number of columns
    cols = 4
    # Define the number of needed rows
    rows = ceil(Int, length(strain_nums) / cols)

    # Initialize figure
    fig = Figure(size=(200 * cols, 200 * rows))
    # Add grid layout
    gl = fig[1, 1] = GridLayout()
    # Loop through metadata
    for (j, strain_num) in enumerate(strain_nums)
        # Extract metadata
        data_meta = df_meta[
            (df_meta.env.==first(data_group.env)).&(df_meta.strain_num.==strain_num),
            :]

        println("   - Plotting geodesic: $(strain_num)")
        # Define row and column index
        row = (j - 1) ÷ cols + 1
        col = (j - 1) % cols + 1
        # Add axis
        ax = Axis(
            gl[row, col],
            aspect=AxisAspect(1),
            title="env: $(first(data_meta.env)) | " *
                  "strain: $(first(data_meta.strain_num))",
            xticksvisible=false,
            yticksvisible=false,
        )
        # Hide axis labels
        hidedecorations!(ax)

        # Plot heatmap of log determinant of metric tensor
        heatmap!(
            ax,
            latent1_range,
            latent2_range,
            logdetG,
            colormap=Reverse(to_colormap(ColorSchemes.PuBu)),
            alpha=0.5,
        )

        # Extract lineage information
        lineage = df_latent[df_latent.strain_num.==data_meta.strain_num, :]

        # Load geodesic state
        geo_state = JLD2.load(first(data_meta.geodesic_state))

        # Generate Brownian bridge
        rnd_bridge = Antibiotic.geometry.brownian_bridge(
            geo_state["latent_init"],
            geo_state["latent_end"],
            length(t_array),
            sigma=sigma,
            num_paths=n_bridges,
            rng=Random.MersenneTwister(42)
        )

        for j in 1:n_bridges
            # Plot Brownian bridge
            lines!(
                ax,
                eachrow(rnd_bridge[:, :, j])...,
                color=(:gray, 0.25),
                linewidth=1.5,
            )
        end # for j in 1:n_bridges

        # Plot lineage
        scatterlines!(
            ax,
            lineage.latent1,
            lineage.latent2,
            color=Antibiotic.viz.colors()[:gold],
            markersize=6,
            linewidth=2,
        )

        # Define NeuralGeodesic model
        nng = NG.NeuralGeodesic(
            nng_template,
            geo_state["latent_init"],
            geo_state["latent_end"],
        )
        # Update model state
        Flux.loadmodel!(nng, geo_state["model_state"])
        # Generate curve
        curve = nng(t_array)
        # Add geodesic line to axis
        lines!(
            ax,
            eachrow(curve)...,
            color=Antibiotic.viz.colors()[:dark_red],
            linewidth=2.5,
        )

        # Add first point 
        scatter!(
            ax,
            [lineage.latent1[1]],
            [lineage.latent2[1]],
            color=Antibiotic.viz.colors()[:dark_red],
            markersize=12,
            marker=:xcross
        )

        # Add last point
        scatter!(
            ax,
            [lineage.latent1[end]],
            [lineage.latent2[end]],
            color=Antibiotic.viz.colors()[:dark_red],
            markersize=12,
            marker=:utriangle
        )
    end # for data_meta in eachrow(chunk)
    # Save figure as PDF
    save("tmp.pdf", fig)
    # Append to final figure
    append_pdf!("$(fname).pdf", "tmp.pdf", cleanup=true)
    # Save figure as PNG
    save("$(fname)_$(first(data_group.env)).png", fig)
end # for chunk in df_meta_chunks

## =============================================================================

println("Plotting specific geodesic trajectories at higher resolution...")

# Initialize figure
fig = Figure(size=(900, 600))

# Add grid layout
gl = fig[1, 1] = GridLayout()

# Define (:env, :strain_num) pairs to plot
pairs = [
    ("KM", 16), ("NFLX", 33), ("TET", 8), ("KM", 28), ("NFLX", 35), ("TET", 3)
]

# Define number of rows and columns
rows = 2
cols = 3

# Loop through pairs
for (i, p) in enumerate(pairs)
    println("env: $(p[1]) | strain: $(p[2])")
    # Extract metadata
    data_meta = df_meta[
        (df_meta.env.==p[1]).&(df_meta.strain_num.==p[2]), :
    ]
    # Define row and column index
    row = (i - 1) ÷ cols + 1
    col = (i - 1) % cols + 1
    # Add axis
    ax = Axis(
        gl[row, col],
        aspect=AxisAspect(1),
        title="evolution antibiotic: $(p[1])",
        xticksvisible=false,
        yticksvisible=false,
    )
    # Hide axis labels
    hidedecorations!(ax)

    # Plot heatmap of log determinant of metric tensor
    hm = heatmap!(
        ax,
        latent1_range,
        latent2_range,
        logdetG,
        colormap=ColorSchemes.tokyo,
    )

    # Plot all points in background
    scatter!(
        ax,
        df_latent.latent1,
        df_latent.latent2,
        markersize=5,
        color=(:gray, 0.25),
        marker=:circle,
    )

    # Extract lineage information
    lineage = df_latent[df_latent.strain_num.==p[2], :]

    # Plot lineage
    scatterlines!(
        ax,
        lineage.latent1,
        lineage.latent2,
        markersize=8,
        linewidth=2,
    )

    # # Load geodesic state
    geo_state = JLD2.load(first(data_meta.geodesic_state))
    # Define NeuralGeodesic model
    nng = NG.NeuralGeodesic(
        nng_template,
        geo_state["latent_init"],
        geo_state["latent_end"],
    )
    # Update model state
    Flux.loadmodel!(nng, geo_state["model_state"])
    # Generate curve
    curve = nng(t_array)
    # Add geodesic line to axis
    lines!(
        ax,
        eachrow(curve)...,
        linewidth=2,
        linestyle=(:dot, :dense),
        color=:white,
    )

    # Add first point 
    scatter!(
        ax,
        [lineage.latent1[1]],
        [lineage.latent2[1]],
        color=:white,
        markersize=18,
        marker=:xcross
    )
    scatter!(
        ax,
        [lineage.latent1[1]],
        [lineage.latent2[1]],
        color=:black,
        markersize=12,
        marker=:xcross
    )

    # Add last point
    scatter!(
        ax,
        [lineage.latent1[end]],
        [lineage.latent2[end]],
        color=:white,
        markersize=18,
        marker=:utriangle
    )
    scatter!(
        ax,
        [lineage.latent1[end]],
        [lineage.latent2[end]],
        color=:black,
        markersize=12,
        marker=:utriangle
    )

    # Check if plot is the last one
    if i == length(pairs)
        # Add grid layout inside grid layout
        gc = gl[1:2, 4] = GridLayout()
        # Add couple of empty grid layouts
        ge_top = gc[1, :] = GridLayout()
        ge_bottom = gc[4, :] = GridLayout()
        # Add colorbar
        Colorbar(gc[2:3, :], hm, label="logdet(G)")
    end
end # for p in pairs

save("$(fig_dir)/geodesic_examples_04.pdf", fig)

fig

## =============================================================================

# Initialize figure
fig = Figure(size=(900, 600))

# Add grid layout
gl = fig[1, 1] = GridLayout()

# Define number of rows and columns
rows = 2
cols = 3

# Loop through pairs
for (i, p) in enumerate(pairs)
    println("env: $(p[1]) | strain: $(p[2])")
    # Extract metadata
    data_meta = df_meta[
        (df_meta.env.==p[1]).&(df_meta.strain_num.==p[2]), :
    ]
    # Define row and column index
    row = (i - 1) ÷ cols + 1
    col = (i - 1) % cols + 1
    # Add axis
    ax = Axis(
        gl[row, col],
        aspect=AxisAspect(1),
        title="evolution antibiotic: $(p[1])",
        xticksvisible=false,
        yticksvisible=false,
    )
    # Hide axis labels
    hidedecorations!(ax)

    # Plot heatmap of log determinant of metric tensor
    hm = heatmap!(
        ax,
        latent1_range,
        latent2_range,
        logdetG,
        colormap=ColorSchemes.tokyo,
    )

    # Plot all points in background
    scatter!(
        ax,
        df_latent.latent1,
        df_latent.latent2,
        markersize=5,
        color=(:gray, 0.25),
        marker=:circle,
    )

    # Extract lineage information
    lineage = df_latent[df_latent.strain_num.==p[2], :]

    # # Load geodesic state
    geo_state = JLD2.load(first(data_meta.geodesic_state))
    # Define NeuralGeodesic model
    nng = NG.NeuralGeodesic(
        nng_template,
        geo_state["latent_init"],
        geo_state["latent_end"],
    )
    # Update model state
    Flux.loadmodel!(nng, geo_state["model_state"])
    # Generate curve
    curve = nng(t_array)
    # Add geodesic line to axis
    lines!(
        ax,
        eachrow(curve)...,
        linewidth=2,
        linestyle=(:dot, :dense),
        color=:white,
    )

    # Add first point 
    scatter!(
        ax,
        [lineage.latent1[1]],
        [lineage.latent2[1]],
        color=:white,
        markersize=18,
        marker=:xcross
    )
    scatter!(
        ax,
        [lineage.latent1[1]],
        [lineage.latent2[1]],
        color=:black,
        markersize=12,
        marker=:xcross
    )

    # Add last point
    scatter!(
        ax,
        [lineage.latent1[end]],
        [lineage.latent2[end]],
        color=:white,
        markersize=18,
        marker=:utriangle
    )
    scatter!(
        ax,
        [lineage.latent1[end]],
        [lineage.latent2[end]],
        color=:black,
        markersize=12,
        marker=:utriangle
    )

    # Check if plot is the last one
    if i == length(pairs)
        # Add grid layout inside grid layout
        gc = gl[1:2, 4] = GridLayout()
        # Add couple of empty grid layouts
        ge_top = gc[1, :] = GridLayout()
        ge_bottom = gc[4, :] = GridLayout()
        # Add colorbar
        Colorbar(gc[2:3, :], hm, label="logdet(G)")
    end
end # for p in pairs

save("$(fig_dir)/geodesic_examples_03.pdf", fig)

fig

## =============================================================================

# Initialize figure
fig = Figure(size=(900, 600))

# Add grid layout
gl = fig[1, 1] = GridLayout()

# Define number of rows and columns
rows = 2
cols = 3

# Loop through pairs
for (i, p) in enumerate(pairs)
    println("env: $(p[1]) | strain: $(p[2])")
    # Extract metadata
    data_meta = df_meta[
        (df_meta.env.==p[1]).&(df_meta.strain_num.==p[2]), :
    ]
    # Define row and column index
    row = (i - 1) ÷ cols + 1
    col = (i - 1) % cols + 1
    # Add axis
    ax = Axis(
        gl[row, col],
        aspect=AxisAspect(1),
        title="evolution antibiotic: $(p[1])",
        xticksvisible=false,
        yticksvisible=false,
    )
    # Hide axis labels
    hidedecorations!(ax)

    # Plot heatmap of log determinant of metric tensor
    hm = heatmap!(
        ax,
        latent1_range,
        latent2_range,
        logdetG,
        colormap=ColorSchemes.tokyo,
    )

    # Plot all points in background
    scatter!(
        ax,
        df_latent.latent1,
        df_latent.latent2,
        markersize=5,
        color=(:gray, 0.25),
        marker=:circle,
    )

    # Extract lineage information
    lineage = df_latent[df_latent.strain_num.==p[2], :]

    # Add first point 
    scatter!(
        ax,
        [lineage.latent1[1]],
        [lineage.latent2[1]],
        color=:white,
        markersize=18,
        marker=:xcross
    )
    scatter!(
        ax,
        [lineage.latent1[1]],
        [lineage.latent2[1]],
        color=:black,
        markersize=12,
        marker=:xcross
    )

    # Add last point
    scatter!(
        ax,
        [lineage.latent1[end]],
        [lineage.latent2[end]],
        color=:white,
        markersize=18,
        marker=:utriangle
    )
    scatter!(
        ax,
        [lineage.latent1[end]],
        [lineage.latent2[end]],
        color=:black,
        markersize=12,
        marker=:utriangle
    )

    # Check if plot is the last one
    if i == length(pairs)
        # Add grid layout inside grid layout
        gc = gl[1:2, 4] = GridLayout()
        # Add couple of empty grid layouts
        ge_top = gc[1, :] = GridLayout()
        ge_bottom = gc[4, :] = GridLayout()
        # Add colorbar
        Colorbar(gc[2:3, :], hm, label="logdet(G)")
    end
end # for p in pairs

save("$(fig_dir)/geodesic_examples_02.pdf", fig)

fig

## =============================================================================

# Initialize figure
fig = Figure(size=(900, 600))

# Add grid layout
gl = fig[1, 1] = GridLayout()

# Define number of rows and columns
rows = 2
cols = 3

# Loop through pairs
for (i, p) in enumerate(pairs)
    println("env: $(p[1]) | strain: $(p[2])")
    # Extract metadata
    data_meta = df_meta[
        (df_meta.env.==p[1]).&(df_meta.strain_num.==p[2]), :
    ]
    # Define row and column index
    row = (i - 1) ÷ cols + 1
    col = (i - 1) % cols + 1
    # Add axis
    ax = Axis(
        gl[row, col],
        aspect=AxisAspect(1),
        title="evolution antibiotic: $(p[1])",
        xticksvisible=false,
        yticksvisible=false,
    )
    # Hide axis labels
    hidedecorations!(ax)

    # Plot heatmap of log determinant of metric tensor
    hm = heatmap!(
        ax,
        latent1_range,
        latent2_range,
        logdetG,
        colormap=ColorSchemes.tokyo,
        alpha=0.0,
    )

    # Plot all points in background
    scatter!(
        ax,
        df_latent.latent1,
        df_latent.latent2,
        markersize=5,
        color=(:gray, 0.25),
        marker=:circle,
    )

    # Extract lineage information
    lineage = df_latent[df_latent.strain_num.==p[2], :]

    # Add first point 
    scatter!(
        ax,
        [lineage.latent1[1]],
        [lineage.latent2[1]],
        color=:white,
        markersize=18,
        marker=:xcross
    )
    scatter!(
        ax,
        [lineage.latent1[1]],
        [lineage.latent2[1]],
        color=:black,
        markersize=12,
        marker=:xcross
    )

    # Add last point
    scatter!(
        ax,
        [lineage.latent1[end]],
        [lineage.latent2[end]],
        color=:white,
        markersize=18,
        marker=:utriangle
    )
    scatter!(
        ax,
        [lineage.latent1[end]],
        [lineage.latent2[end]],
        color=:black,
        markersize=12,
        marker=:utriangle
    )

    # Check if plot is the last one
    if i == length(pairs)
        # Add grid layout inside grid layout
        gc = gl[1:2, 4] = GridLayout()
        # Add couple of empty grid layouts
        ge_top = gc[1, :] = GridLayout()
        ge_bottom = gc[4, :] = GridLayout()
        # Add colorbar
        Colorbar(gc[2:3, :], hm, label="logdet(G)")
    end
end # for p in pairs

save("$(fig_dir)/geodesic_examples_01.pdf", fig)

fig

## =============================================================================

# Initialize figure
fig = Figure(size=(900, 600))

# Add grid layout
gl = fig[1, 1] = GridLayout()

# Define number of rows and columns
rows = 2
cols = 3

# Loop through pairs
for (i, p) in enumerate(pairs)
    println("env: $(p[1]) | strain: $(p[2])")
    # Extract metadata
    data_meta = df_meta[
        (df_meta.env.==p[1]).&(df_meta.strain_num.==p[2]), :
    ]
    # Define row and column index
    row = (i - 1) ÷ cols + 1
    col = (i - 1) % cols + 1
    # Add axis
    ax = Axis(
        gl[row, col],
        aspect=AxisAspect(1),
        title="evolution antibiotic: $(p[1])",
        xticksvisible=false,
        yticksvisible=false,
    )
    # Hide axis labels
    hidedecorations!(ax)

    # Plot heatmap of log determinant of metric tensor
    hm = heatmap!(
        ax,
        latent1_range,
        latent2_range,
        logdetG,
        colormap=ColorSchemes.tokyo,
        alpha=0.0,
    )

    # Plot all points in background
    scatter!(
        ax,
        df_latent.latent1,
        df_latent.latent2,
        markersize=5,
        color=(:gray, 0.25),
        marker=:circle,
    )

    # Extract lineage information
    lineage = df_latent[df_latent.strain_num.==p[2], :]

    # Check if plot is the last one
    if i == length(pairs)
        # Add grid layout inside grid layout
        gc = gl[1:2, 4] = GridLayout()
        # Add couple of empty grid layouts
        ge_top = gc[1, :] = GridLayout()
        ge_bottom = gc[4, :] = GridLayout()
        # Add colorbar
        Colorbar(gc[2:3, :], hm, label="logdet(G)")
    end
end # for p in pairs

save("$(fig_dir)/geodesic_examples_00.pdf", fig)

fig

## =============================================================================

# Define (:env, :strain_num) pairs to plot
pairs = [
    ("KM", 16), ("NFLX", 33), ("TET", 8), ("KM", 28), ("NFLX", 35), ("TET", 3)
]

# Define filename
fname = "$(fig_dir)/resistance_trajectories_example.pdf"

# If file exists, delete it
if isfile(fname)
    rm(fname)
end

# Loop through pairs
for p in pairs
    println("Plotting resistance trajectory: $(p[1]) | strain #$(p[2])")
    # Initialize figure
    fig = Figure(size=(900, 500))

    # Add grid layout
    gl = fig[1, 1] = GridLayout()

    # Add grid layour for subplots
    gl_plots = gl[1, 1] = GridLayout()
    # Add grid layout for legend
    gl_legend = gl[2, 1] = GridLayout()

    # Define number of rows and columns
    rows = 2
    cols = 4


    # Extract geodesic metadata
    data_meta = df_meta[
        (df_meta.env.==p[1]).&(df_meta.strain_num.==p[2]), :
    ]
    # Load geodesic state
    geo_state = JLD2.load(first(data_meta.geodesic_state))
    # Define NeuralGeodesic model
    nng = NG.NeuralGeodesic(
        nng_template,
        geo_state["latent_init"],
        geo_state["latent_end"],
    )
    # Update model state
    Flux.loadmodel!(nng, geo_state["model_state"])
    # Generate curve
    curve = nng(t_array)
    # Run curve through decoder
    curve_decoded = rhvae.vae.decoder(curve).μ

    # Define linear interpolation between beginning and end of curve
    line = Antibiotic.geometry.linear_interpolation(
        geo_state["latent_init"],
        geo_state["latent_end"],
        length(t_array)
    )
    # Run line through decoder
    line_decoded = rhvae.vae.decoder(line).μ

    # Loop through drugs
    for (i, drug) in enumerate(drug_list)
        # Define row and column index
        row = (i - 1) ÷ cols + 1
        col = (i - 1) % cols + 1
        # Add axis
        ax = Axis(
            gl_plots[row, col],
            aspect=AxisAspect(1.25),
            title="$(drug)",
            xlabel="days",
            ylabel="logIC₅₀",
        )
        # Extract strain information
        data_strain = df_logic50[
            (df_logic50.drug.==drug).&(df_logic50.strain_num.==p[2]), :]
        # Plot strain information
        scatterlines!(
            ax,
            data_strain.day,
            data_strain.logic50_mean_std,
        )
        # Add error bars
        errorbars!(
            ax,
            data_strain.day,
            data_strain.logic50_mean_std,
            data_strain.logic50_mean_std .- data_strain.logic50_ci_lower_std,
            data_strain.logic50_ci_upper_std .- data_strain.logic50_mean_std,
            label="experimental",
        )

        # Plot decoded curve
        # Create scaled x-axis from min to max day values
        scaled_days = range(
            minimum(data_strain.day),
            maximum(data_strain.day),
            length=length(curve_decoded[i, :])
        )

        # Plot decoded curve with scaled days
        lines!(
            ax,
            scaled_days,
            curve_decoded[i, :],
            color=Antibiotic.viz.colors()[:dark_red],
            label="RHVAE geodesic",
            linewidth=2,
        )

        # Plot linear interpolation
        lines!(
            ax,
            scaled_days,
            line_decoded[i, :],
            color=Antibiotic.viz.colors()[:gold],
            label="linear interpolation",
            linewidth=2,
        )

        # Add legend if first plot
        if i == 1
            Legend(
                gl_legend[1, 1],
                ax,
                orientation=:horizontal,
                merge=true,
                framevisible=false,
            )
        end # if i == 1
    end # for drug in drug_list

    # Add title
    Label(
        gl[1, 1, Top()],
        "$(p[1]) | strain #$(p[2])",
        fontsize=20,
        padding=(0, 0, 30, 0),
    )

    # Change spacing between subplots
    rowgap!(gl_plots, -20)

    # Change spacing between legend and subplots
    rowgap!(gl, -10)

    # Save figure
    save("tmp.pdf", fig)
    # Append to final figure
    append_pdf!(fname, "tmp.pdf", cleanup=true)
end

## =============================================================================

# Define filename
fname = "$(fig_dir)/resistance_trajectories.pdf"

# If file exists, delete it
if isfile(fname)
    rm(fname)
end

# Group df_meta by env and strain_num
df_group = DF.groupby(df_meta, [:env, :strain_num])

# Loop through groups
for data_meta in df_group
    println("Plotting resistance trajectory: $(data_meta.env[1]) | strain #$(data_meta.strain_num[1])")
    # Initialize figure
    fig = Figure(size=(900, 500))

    # Add grid layout
    gl = fig[1, 1] = GridLayout()

    # Add grid layour for subplots
    gl_plots = gl[1, 1] = GridLayout()
    # Add grid layout for legend
    gl_legend = gl[2, 1] = GridLayout()

    # Define number of rows and columns
    rows = 2
    cols = 4

    # Extract geodesic metadata
    geo_state = JLD2.load(first(data_meta.geodesic_state))
    # Define NeuralGeodesic model
    nng = NG.NeuralGeodesic(
        nng_template,
        geo_state["latent_init"],
        geo_state["latent_end"],
    )
    # Update model state
    Flux.loadmodel!(nng, geo_state["model_state"])
    # Generate curve
    curve = nng(t_array)
    # Run curve through decoder
    curve_decoded = rhvae.vae.decoder(curve).μ

    # Define linear interpolation between beginning and end of curve
    line = Antibiotic.geometry.linear_interpolation(
        geo_state["latent_init"],
        geo_state["latent_end"],
        length(t_array)
    )
    # Run line through decoder
    line_decoded = rhvae.vae.decoder(line).μ

    # Loop through drugs
    for (i, drug) in enumerate(drug_list)
        # Define row and column index
        row = (i - 1) ÷ cols + 1
        col = (i - 1) % cols + 1
        # Add axis
        ax = Axis(
            gl_plots[row, col],
            aspect=AxisAspect(1.25),
            title="$(drug)",
            xlabel="days",
            ylabel="logIC₅₀",
        )
        # Extract strain information
        data_strain = df_logic50[
            (df_logic50.drug.==drug).&(df_logic50.strain_num.==data_meta.strain_num[1]), :]
        # Plot strain information
        scatterlines!(
            ax,
            data_strain.day,
            data_strain.logic50_mean_std,
            linewidth=2.5,
        )
        # Add error bars
        errorbars!(
            ax,
            data_strain.day,
            data_strain.logic50_mean_std,
            data_strain.logic50_mean_std .- data_strain.logic50_ci_lower_std,
            data_strain.logic50_ci_upper_std .- data_strain.logic50_mean_std,
            label="experimental",
        )

        # Plot decoded curve
        # Create scaled x-axis from min to max day values
        scaled_days = range(
            minimum(data_strain.day),
            maximum(data_strain.day),
            length=length(curve_decoded[i, :])
        )

        # Plot decoded curve with scaled days
        lines!(
            ax,
            scaled_days,
            curve_decoded[i, :],
            color=Antibiotic.viz.colors()[:dark_red],
            label="RHVAE geodesic",
            linewidth=2.5,
        )

        # Plot linear interpolation
        lines!(
            ax,
            scaled_days,
            line_decoded[i, :],
            color=Antibiotic.viz.colors()[:gold],
            label="linear interpolation",
            linewidth=2.5,
        )

        # Add legend if first plot
        if i == 1
            Legend(
                gl_legend[1, 1],
                ax,
                orientation=:horizontal,
                merge=true,
                framevisible=false,
            )
        end # if i == 1
    end # for drug in drug_list

    # Add title
    Label(
        gl[1, 1, Top()],
        "$(data_meta.env[1]) | strain #$(data_meta.strain_num[1])",
        fontsize=20,
        padding=(0, 0, 30, 0),
    )

    # Change spacing between subplots
    rowgap!(gl_plots, -20)

    # Change spacing between legend and subplots
    rowgap!(gl, -10)

    # Save figure
    save("tmp.pdf", fig)
    # Append to final figure
    append_pdf!(fname, "tmp.pdf", cleanup=true)
end