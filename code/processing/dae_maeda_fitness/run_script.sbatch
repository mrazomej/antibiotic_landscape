#!/bin/bash

#SBATCH --time=12:00:00 			            # time to run task
#SBATCH --job-name=dae_fitness                  # task name
#SBATCH --partition=normal,hns,dpetrov,owners	# partition
#SBATCH --nodes=1				                # node count
#SBATCH --ntasks=1				                # total number of tasks across all nodes
#SBATCH --cpus-per-task=1			            # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=10G			            # memory per cpu-core (4G is default)
#SBATCH --gpus-per-task=1			            # gpus
#SBATCH --mail-type=begin			            # send email when job begins
#SBATCH --mail-type=end				            # send email when job ends
#SBATCH --mail-type=fail			            # send email if job fails
#SBATCH --output=date.out			            # filename where to save output

# Load CUDA
# ml cuda
# load CUDA-related libraries
ml cudnn/8.1.1.33 cutensor/1.5.0.3

# Load julia
ml julia

# Set debugger
export JULIA_DEBUG=CUDA
export JULIA_CUDA_USE_BINARYBUILDER=false

# Launch julia and run script
julia daegpu_batch.jl 
