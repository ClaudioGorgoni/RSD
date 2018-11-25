#!/bin/bash

#SBATCH -p p4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 02:00:00
#SBATCH -J mcmc
#SBATCH -o output/elg_sgc_bin5_nopeak/rsd_mcmc.stdout
#SBATCH -e output/elg_sgc_bin5_nopeak/rsd_mcmc.stderr

module unload mpi4py

export OMP_NUM_THREADS=1
export NUM_THREADSPROCESSES=${SLURM_CPUS_PER_TASK}

./main.py