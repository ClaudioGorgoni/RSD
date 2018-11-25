#!/bin/bash

#SBATCH -p s2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --exclusive
#SBATCH -J pso
#SBATCH -o output/elg_all_bin8/pso.stdout
#SBATCH -e output/elg_all_bin8/pso.stderr

module unload mpi4py

export OMP_NUM_THREADS=1
export NUM_THREADSPROCESSES=${SLURM_CPUS_PER_TASK}

./pso_minimizer.py