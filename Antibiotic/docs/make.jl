using Documenter
using Antibiotic

makedocs(
    sitename = "Antibiotic",
    format = Documenter.HTML(),
    modules = [Antibiotic]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
