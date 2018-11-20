from os.path import exists
from os import makedirs
import ConfigParser
import os

config = ConfigParser.ConfigParser()
config.read("par.ini")

label  = config.get('mcmc','label')
output = 'output/'+label
if not exists(output):
	makedirs(output)

tmp = open('pso.slurm', 'w')

content = ("#!/bin/bash\n\n"

           "#SBATCH -p p4\n"
           "#SBATCH --nodes=1\n"
           "#SBATCH --ntasks-per-node=1\n"
           "#SBATCH --cpus-per-task=16\n"
           "#SBATCH --exclusive\n"
           "#SBATCH -J pso\n"
           "#SBATCH -o " + output + "/pso.stdout\n"
           "#SBATCH -e " + output + "/pso.stderr\n\n"

           "module unload mpi4py\n\n"
    
           "export OMP_NUM_THREADS=1\n"
           "export NUM_THREADSPROCESSES=${SLURM_CPUS_PER_TASK}\n\n"
    
           "./pso_minimizer.py"
           )
tmp.write(content)
tmp.close()