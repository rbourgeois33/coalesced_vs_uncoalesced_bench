#!/bin/bash
#SBATCH -A pri@a100

#SBATCH --job-name=some_timing_kokkos        # name of job

# Other partitions are usable by activating/uncommenting
# one of the 5 following directives:
##SBATCH -C v100-16g                 # uncomment to target only 16GB V100 GPU
##SBATCH -C v100-32g                 # uncomment to target only 32GB V100 GPU
##SBATCH --partition=gpu_p2          # uncomment for gpu_p2 partition (32GB V100 GPU)
#SBATCH -C a100                     # uncomment for gpu_p5 partition (80GB A100 GPU)
##SBATCH -C h100                     # uncomment for gpu_p6 partition (80GB H100 GPU)
#SBATCH --exclusive
# Here, reservation of 10 CPUs (for 1 task) and 1 GPU on a single node:
#SBATCH --nodes=1                    # we request one node
#SBATCH --ntasks-per-node=1          # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:1                 # number of GPUs per node (max 8 with gpu_p2, gpu_p5)

# The number of CPUs per task must be adapted according to the partition used. Knowing that here
# only one GPU is reserved (i.e. 1/4 or 1/8 of the GPUs of the node depending on the partition),
# the ideal is to reserve 1/4 or 1/8 of the CPUs of the node for the single task:
##SBATCH --cpus-per-task=10           # number of cores per task (1/4 of the 4-GPUs V100 node)
##SBATCH --cpus-per-task=3           # number of cores per task for gpu_p2 (1/8 of 8-GPUs V100 node)
#SBATCH --cpus-per-task=8           # number of cores per task for gpu_p5 (1/8 of 8-GPUs A100 node)
##SBATCH --cpus-per-task=24          # number of cores per task for gpu_p6 (1/4 of 4-GPUs H100 node)

# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=00:10:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=some_timing_kokkos%j.out    # name of output file
#SBATCH --error=some_timing_kokkos%j.out     # name of error file (here, in common with the output file)

# Cleans out the modules loaded in interactive and inherited by default 
module purge

# Uncomment the following module command if you are using the "gpu_p5" partition
# to have access to the modules compatible with this partition.
module load arch/a100
# Uncomment the following module command if you are using the "gpu_p6" partition
# to have access to the modules compatible with this partition.
#module load arch/h100

# Loading of modules
module load  cuda/12.8.0
module load python/3.9.12
# Echo of launched commands
set -x

# For the "gpu_p5" and "gpu_p6" partitions, the code must be compiled with the modules compatible
# with the choosen partition.
# Code execution
./../run-bench.sh
