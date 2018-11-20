#!/bin/bash

#SBATCH -p p4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH -J test
#SBATCH -o test.stdout
#SBATCH -e test.stderr

module unload mpi4py

export OMP_NUM_THREADS=1
export NUM_THREADSPROCESSES=${SLURM_CPUS_PER_TASK}

./test.py
