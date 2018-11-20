#!/bin/bash

#SBATCH -p s1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --exclusive
#SBATCH -J mcmc
#SBATCH -o output/mock_sgc/rsd_mcmc.stdout
#SBATCH -e output/mock_sgc/rsd_mcmc.stderr

module unload mpi4py

export OMP_NUM_THREADS=1
export NUM_THREADSPROCESSES=${SLURM_CPUS_PER_TASK}

./main.py