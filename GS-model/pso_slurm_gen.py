from os.path import exists
from os import makedirs
import ConfigParser
import os

config = ConfigParser.ConfigParser()
config.read("par.ini")

label  = config.get('mcmc','label')
queue  = config.get('slurm', 'queue')
n      = config.getint('slurm', 'ncpus')

output = 'output/'+label
if not exists(output):
	makedirs(output)

tmp = open('pso.slurm', 'w')

content = ("#!/bin/bash\n\n"

           "#SBATCH -p "+queue+"\n"
           "#SBATCH --nodes=1\n"
           "#SBATCH --ntasks-per-node=1\n"
           "#SBATCH --cpus-per-task="+str(n)+"\n"
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
