#!/bin/bash

#SBATCH -p p4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH -J pso
#SBATCH -o output/elg_sgc/pso.stdout
#SBATCH -e output/elg_sgc/pso.stderr

module unload mpi4py

export OMP_NUM_THREADS=1
export NUM_THREADSPROCESSES=${SLURM_CPUS_PER_TASK}

./pso_minimizer.py