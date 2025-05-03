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

# Define output directory
fig_dir = "$(git_root())/fig/supplementary"

# Generate output directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating output directory...")
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

# Define number of rows and columns
rows = 2
cols = 4

# Loop through each drug
for evo_drug in evo_drugs
    println("Plotting IC50 profiles for $(evo_drug)...")

    # Find index of entries where :env contain "in_$(evo_drug)"
    idx = findall(x -> occursin("in_$(evo_drug)", x), df.env)

    # Group data by drug
    df_group = DF.groupby(df[idx, :], :drug)

    # Convert GroupedDataFrame to Vector and reorder so matching drug comes first
    df_group_vec = collect(df_group)
    sort!(df_group_vec, by=x -> x[1, :drug] != evo_drug)

    # --------------------------------------------------------------------------
    # Initialize figure 
    fig = Figure(size=(200 * cols, 200 * rows))

    # Add global grid layout
    gl = fig[1, 1] = GridLayout()

    # Add grid layout for banner
    gl_banner = gl[1, 1] = GridLayout()

    # Add grid layout for plot
    gl_plot = gl[2, 1] = GridLayout()

    # --------------------------------------------------------------------------
    # Add banner

    # Add box for section title
    Box(
        gl_banner[1, 1],
        color="#E6E6EF",
        strokecolor="#E6E6EF",
        alignmode=Mixed(; left=-50, right=-50) # Moves box to the left and right
    )

    # Add section title
    Label(
        gl_banner[1, 1],
        "IC₅₀ profiles for strains evolved in $(evo_drug)",
        fontsize=14,
        padding=(0, 0, 0, 0),
        halign=:left,
        tellwidth=false, # prevent column from contracting because of label size
        alignmode=Mixed(; left=-30) # Moves text to the left
    )

    # --------------------------------------------------------------------------

    # Define colors for each strain as a dictionary
    colors = Dict(
        sort(unique(df[idx, :strain_num])) .=>
            ColorSchemes.glasbey_hv_n256[1:length(unique(df[idx, :strain_num]))]
    )

    # Loop through each drug
    for (i, data) in enumerate(df_group_vec)
        # Define index for row and column
        row = (i - 1) ÷ cols + 1
        col = (i - 1) % cols + 1
        # Add axis
        ax = Axis(
            gl_plot[row, col],
            xlabel="day",
            ylabel="log(IC₅₀)",
            title=(data[1, :drug] == evo_drug) ?
                  "$(data[1, :drug]) (selection)" :
                  "$(data[1, :drug])",
            titlesize=12,
            xlabelsize=12,
            ylabelsize=12,
            xticklabelsize=12,
            yticklabelsize=12,
            aspect=AxisAspect(4 / 3),
        )

        # Group data by :strain_num
        data_group = DF.groupby(data, [:strain_num])

        # Loop through each strain
        for (j, strain) in enumerate(data_group)
            # Sort data by day
            DF.sort!(strain, :day)
            # Extract strain number
            strain_num = first(strain.strain_num)
            # Plot data
            scatterlines!(
                strain.day,
                strain.logic50_mean,
                color=colors[strain_num],
                markersize=6,
            )
        end # for
    end # for

    # Adjust gap between rows and columns
    rowgap!(gl_plot, 5)
    colgap!(gl_plot, 5)

    # Save figure
    save("$(fig_dir)/figSI_ic50_profiles_$(evo_drug).pdf", fig)
    save("$(fig_dir)/figSI_ic50_profiles_$(evo_drug).png", fig)


end # for
