#!/bin/bash
#SBATCH --account=cad15920
#SBATCH --job-name="kokkos_hip"
#SBATCH --constraint=MI300
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=1:00:00
#SBATCH --output=kokkos_hip_%j.out
#SBATCH --error=kokkos_hip_%j.err
#SBATCH --exclusive 

# Clean environment
module purge

# Load required modules
module load cpe/24.07
module load craype-accel-amd-gfx942 craype-x86-trento
module load PrgEnv-cray
module load amd-mixed
module list

module list
# For <= 4 APUs (single node)
export NCCL_MIN_NCHANNELS=42
export NCCL_NSOCKS_PERTHREAD=1
export NCCL_SOCKET_NTHREADS=1
export HSA_XNACK=1
#Run the executable
# Adjust the executable name and arguments as needed
./../run-bench.sh
