# Learning the Shape of Evolutionary Landscapes: Geometric Deep Learning Reveals Hidden Structure in Phenotype-to-Fitness Maps

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia](https://img.shields.io/badge/Julia-1.9+-blue.svg)](https://julialang.org/)

This repository contains the code and data for the research project on learning
evolutionary landscapes using geometric deep learning methods. The project
demonstrates how Riemannian Hamiltonian Variational Autoencoders (RHVAE) can
uncover hidden structure in high-dimensional fitness data and provide insights
into evolutionary dynamics.

## 📖 Publication

- **Paper**: [Learning the Shape of Evolutionary Landscapes: Geometric Deep Learning Reveals Hidden Structure in Phenotype-to-Fitness Maps](https://mrazomej.github.io/antibiotic_landscape/paper.html)
- **Preprint**: [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.05.07.652616v1)

## 🎯 Overview

This project addresses the fundamental challenge in evolutionary biology of
understanding the complex relationships between genotypes, phenotypes, and
fitness. We present a computational framework that:

1. **Learns low-dimensional representations** of high-dimensional fitness
   profiles using geometry-informed variational autoencoders
2. **Captures nonlinear structure** in phenotype-fitness maps that linear
   methods miss
3. **Provides geometric information** about the learned latent space for
   meaningful interpretation
4. **Demonstrates superior predictive power** for out-of-sample data compared to
   traditional approaches

The method is validated on simulated adaptive dynamics and applied to real
antibiotic resistance data from *E. coli*.

## 🏗️ Repository Structure

```
antibiotic_landscape/
├── Antibiotic/                    # Main Julia module
│   ├── src/
│   │   ├── Antibiotic.jl         # Module entry point
│   │   ├── geometry.jl           # Geometric analysis functions
│   │   ├── metropolis.jl         # Metropolis-Hastings evolution
│   │   ├── stats.jl              # Statistical utilities
│   │   └── viz.jl                # Visualization functions
│   └── docs/                     # Documentation
├── code/                         # Analysis and processing code
│   ├── analysis/                 # Data analysis scripts
│   ├── exploratory/              # Exploratory notebooks
│   ├── fig/                      # Figure generation scripts
│   └── processing/               # Data processing pipelines
│       ├── beta-rhvae_*/         # RHVAE training and analysis
│       ├── kinsler_2020/         # Kinsler dataset processing
│       └── mcmc_iwasawa_logistic/ # MCMC analysis
├── data/                         # Raw and processed data
├── paper/                        # Manuscript and figures
│   ├── fig/                      # Generated figures
│   ├── code/                     # Paper-specific code
│   └── *.qmd                     # Quarto manuscript files
└── docs/                         # Documentation website
```

## 🚀 Quick Start

### Prerequisites

- Julia 1.9 or higher
- Required Julia packages (see `Antibiotic/Project.toml`)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mrazomej/antibiotic_landscape.git
cd antibiotic_landscape
```

2. Install Julia dependencies:
```bash
cd Antibiotic
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

3. Activate the environment and run examples:
```julia
using Pkg
Pkg.activate("Antibiotic")
using Antibiotic
```

## 📊 Key Components

### 1. Antibiotic Module (`Antibiotic/`)

The main Julia module providing core functionality:

- **`geometry.jl`**: Procrustes analysis, Fréchet distance calculations, and
  geometric transformations
- **`metropolis.jl`**: Metropolis-Hastings evolutionary dynamics simulation with
  Gaussian fitness landscapes
- **`stats.jl`**: Statistical utilities for data analysis
- **`viz.jl`**: Visualization functions for results

### 2. Processing Pipelines (`code/processing/`)

Organized by model type and version:

- **`beta-rhvae_jointlogencoder_simpledecoder_iwasawa_mcmc/`**: RHVAE training
  and analysis on Iwasawa dataset
- **`kinsler_2020/`**: Processing for Kinsler et al. fitness data
- **`mcmc_iwasawa_logistic/`**: MCMC analysis of IC₅₀ values

### 3. Analysis Scripts (`code/analysis/`)

- **`geodesic_plots.jl`**: Visualization of geodesic paths in latent space
- **`pca_vs_geodesics_plots.jl`**: Comparison of PCA vs. geometric methods

### 4. Exploratory Notebooks (`code/exploratory/`)

Jupyter notebooks for:
- Gradient ascent analysis
- Bayesian inference
- Differential geometry exploration
- Legacy experiments

## 🔬 Core Methods

### Riemannian Hamiltonian Variational Autoencoder (RHVAE)

The main innovation of this work is the use of geometry-informed variational
autoencoders that:

- Learn a low-dimensional latent representation of high-dimensional fitness data
- Preserve geometric structure through a learned metric tensor
- Enable meaningful distance calculations in the latent space
- Provide superior reconstruction and prediction compared to linear methods

### Metropolis-Hastings Evolution

Simulated evolutionary dynamics using:
- Gaussian fitness landscapes with multiple peaks
- Metropolis-Hastings sampling for population evolution
- Environment-specific fitness landscapes
- Trajectory analysis and comparison

### Geometric Analysis

Tools for analyzing the learned latent space:
- Procrustes analysis for alignment
- Fréchet distance for trajectory comparison
- Geodesic path calculation
- Metric tensor visualization

## 📈 Results

The project demonstrates:

1. **Superior dimensionality reduction**: 2D RHVAE achieves reconstruction
   accuracy comparable to 5D PCA
2. **Better out-of-sample prediction**: Nonlinear latent space more accurately
   predicts antibiotic resistance profiles
3. **Geometric interpretability**: Learned metric provides meaningful distance
   measures
4. **Evolutionary insights**: Captures constraints and predictability in
   antibiotic resistance evolution

## 🧪 Data

The analysis uses:
- **Iwasawa et al. (2022)**: High-throughput fitness measurements of *E. coli*
  under different antibiotic pressures
- **Simulated data**: Metropolis-Hastings evolution on Gaussian fitness
  landscapes
- **Cross-validation**: Out-of-sample prediction testing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.

## 🙏 Acknowledgments

We thank:
- David Larios, Enrique Amaya, and Griffin Chure for helpful discussions
- Jose Aguilar-Rodriguez, Stefan Bassler, Benjamin Good, and others for
  manuscript feedback
- Junichiro Iwasawa for sharing the raw data from his study

## 📚 References

1. Iwasawa, J., et al. (2022). Analysis of the evolution of resistance to
   multiple antibiotics enables prediction of the Escherichia coli
   phenotype-based fitness landscape. *PLOS Biology*, 20(12), e3001920.

2. Chadebec, C., Mantoux, C., & Allassonnière, S. (2020). Geometry-Aware
   Hamiltonian Variational Auto-Encoder. *arXiv preprint* arXiv:2010.11518.

3. Kinsler, G., Geiler-Samerotte, K., & Petrov, D. A. (2020). Fitness variation
   across subtle environmental perturbations reveals local modularity and global
   pleiotropy of adaptation. *eLife*, 9, e61271.

## 📞 Contact

- **Manuel Razo-Mejia**: Department of Biology, Stanford University
- **Madhav Mani**: NSF-Simons Center for Quantitative Biology, Northwestern
  University
- **Dmitri A. Petrov**: Department of Biology, Stanford University

For questions about the code or methodology, please open an issue on GitHub.