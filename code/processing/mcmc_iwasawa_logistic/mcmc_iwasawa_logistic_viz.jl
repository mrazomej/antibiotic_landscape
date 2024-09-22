## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic

# Import package to handle DataFrames
import DataFrames as DF
import CSV

# Import Glob to list files
import Glob

# Load CairoMakie for plotting
using CairoMakie
import PairPlots
import ColorSchemes

# Import basic math libraries
import StatsBase
import LinearAlgebra
import Random

# Activate backend
CairoMakie.activate!()

# Set PBoC Plotting style
Antibiotic.viz.theme_makie!()

## =============================================================================

# Locate current directory
path_dir = pwd()
# Find the path prefix where to put figures
path_prefix = replace(
    match(r"processing/(.*)", path_dir).match,
    "processing" => "",
)

# Define figure directory
fig_dir = "$(git_root())/fig$(path_prefix)"

# Create figure directory if it does not exist
if !isdir(fig_dir)
    println("Creating figure directory...")
    mkpath(fig_dir)
end

## =============================================================================

println("Loading MCMC summary results...")

# Load results
df = CSV.File(
    "$(git_root())/output/mcmc_iwasawa_logistic/logic50_ci.csv"
) |> DF.DataFrame

## =============================================================================

println("Plotting IC50 profiles...")

# Define the drugs in which strains evolved
evo_drugs = unique([split(x, "_")[end] for x in df.env])

# Loop through each drug
for evo_drug in evo_drugs
    println("Plotting IC50 profiles for $(evo_drug)...")

    # Find index of entries where :env contain "in_$(evo_drug)"
    idx = findall(x -> occursin("in_$(evo_drug)", x), df.env)

    # Group data by drug
    df_group = DF.groupby(df[idx, :], :drug)

    # Initialize figure 
    fig = Figure(size=(900, 500))

    # Add grid layout
    gl = fig[1, 1] = GridLayout()

    # Define number of rows and columns
    rows = 2
    cols = 4

    # Loop through each drug
    for (i, data) in enumerate(df_group)
        # Define index for row and column
        row = (i - 1) ÷ cols + 1
        col = (i - 1) % cols + 1
        # Add axis
        ax = Axis(
            gl[row, col],
            xlabel="day",
            ylabel="log(IC50)",
            aspect=AxisAspect(1),
            title="$(data[1, :drug])",
        )

        # Group data by :strain_num
        data_group = DF.groupby(data, [:strain_num])

        # Define colors
        colors = get(ColorSchemes.inferno, range(0.0, 0.5, length(data_group)))

        # Loop through each strain
        for (j, strain) in enumerate(data_group)
            # Sort data by day
            DF.sort!(strain, :day)
            # Plot data
            scatterlines!(
                strain.day,
                strain.logic50_mean,
                # color=ColorSchemes.tab20[j],
                color=colors[j],
            )
        end # for
    end # for

    # Add global title
    Label(
        gl[1, :, Top()],
        "Evolution environment: $(evo_drug)",
        valign=:bottom,
        font=:bold,
        padding=(0, 0, 30, 0),
        fontsize=20,
    )

    # Save figure
    save("$(fig_dir)/ic50_profiles_$(evo_drug).pdf", fig)

    fig

end # for

## =============================================================================

println("Read the raw data...")

# Load data into a DataFrame
df = CSV.read(
    "$(git_root())/data/Iwasawa_2022/iwasawa_tidy.csv", DF.DataFrame
)

# Remove blank measurements
df = df[.!df.blank, :]
# Remove zero concentrations
df = df[df.concentration_ugmL.>0, :]

## =============================================================================

# Define data to use
data = df[(df.antibiotic.=="KM").&(df.env.=="Parent_in_KM").&(df.strain_num.==13).&.!(df.blank).&(df.concentration_ugmL.>0), :]
# Remove blank measurement
# Group data by day
df_group = DF.groupby(data, :day)

# Initialize figure
fig = Figure(size=(500, 300))

# Add grid layout
gl = fig[1, 1] = GridLayout()

# Add axis
ax = Axis(
    gl[1, 1],
    xlabel="antibiotic concentration",
    ylabel="OD₆₂₀",
    xscale=log10
)

# Define colors for plot
colors = get(ColorSchemes.Blues_9, LinRange(0.25, 1, length(df_group)))

# Loop through days
for (i, d) in enumerate(df_group)
    # Sort data by concentration
    DF.sort!(d, :concentration_ugmL)
    # Plot scatter line
    scatterlines!(
        ax, d.concentration_ugmL, d.OD, color=colors[i], label="$(first(d.day))"
    )
end # for

# Add legend to plot
gl[1, 2] = Legend(
    fig, ax, "day", framevisible=false, nbanks=3, labelsize=10
)

save("$(fig_dir)/example_data.pdf", fig)

fig

## =============================================================================

println("Define logistic function fit to data...")

@doc raw"""
    logistic(x, a, b, c, ic50)

Compute the logistic function used to model the relationship between antibiotic
concentration and bacterial growth.

This function implements the following equation:

f(x) = a / (1 + (x / ic50)^b) + c

# Arguments
- `x`: Antibiotic concentration (input variable)
- `a`: Maximum effect parameter (difference between upper and lower asymptotes)
- `b`: Slope parameter (steepness of the curve)
- `c`: Minimum effect parameter (lower asymptote)
- `ic50`: IC₅₀ parameter (concentration at which the effect is halfway between
  the minimum and maximum)

# Returns
The computed effect (e.g., optical density) for the given antibiotic
concentration and parameters.

Note: This function is vectorized and can handle array inputs for `x`.
"""
function logistic(logx, a, b, c, logic50)
    return @. a / (1.0 + exp(b * (logx - logic50))) + c
end

## =============================================================================

println("List MCMC samples...")

# Define output directory
out_dir = "$(git_root())/data/Iwasawa_2022/mcmc_nonnegative"

# List all files in the output directory
files = sort(Glob.glob("$(out_dir)/*.csv"[2:end], "/"))

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

println("Plot example fits...")

Random.seed!(42)

# Initialize figure
fig = Figure(size=(900, 500))

# Define number of rows and columns
rows = 2
cols = 4

# Select random indexes
idxs = Random.randperm(size(df_meta, 1))[1:rows*cols]

# Loop through each plot
for i in 1:rows*cols
    # Locate row and column index
    row_idx = (i - 1) ÷ cols + 1
    col_idx = (i - 1) % cols + 1

    # Extract row to analyze
    row = df_meta[idxs[i], :]

    # Add axis
    ax = Axis(
        fig[row_idx, col_idx],
        xlabel="antibiotic concentration",
        ylabel="optical density",
        title="$(replace(row.env, "_" => " ")) | day $(row.day)",
        xscale=log10,
        yscale=log10,
    )

    # Load the file
    chain = CSV.read(row.file, DF.DataFrame)

    # Extract corresponding data from the raw data
    data = df[
        (df.antibiotic.==row.antibiotic).&(df.day.==row.day).&(df.env.==row.env).&(df.strain_num.==row.strain_num),
        :]
    # Sort data by concentration
    sort!(data, :concentration_ugmL)

    # Locate unique concentrations
    unique_concentrations = unique(data.concentration_ugmL)

    # Initialize matrix to store samples
    y_samples = Array{Float64}(
        undef, length(unique_concentrations), size(chain, 1)
    )

    # Loop through samples
    for i in 1:size(chain, 1)
        logy_samples = log.(logistic(
            log.(unique_concentrations),
            chain[i, :a],
            chain[i, :b],
            chain[i, :c],
            chain[i, :logic50]
        ))
        # Add noise
        logy_samples .+= randn(length(unique_concentrations)) * √(chain[i, :σ²])
        y_samples[:, i] = exp.(logy_samples)

        # Plot samples
        lines!(
            ax,
            unique_concentrations,
            y_samples[:, i],
            color=(ColorSchemes.Paired_12[1], 0.5)
        )
    end # for

    # Plot data
    scatter!(
        ax, data.concentration_ugmL, data.OD, color=ColorSchemes.Paired_12[2]
    )

end # for

fig