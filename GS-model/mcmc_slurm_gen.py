import ConfigParser
import os
from os import makedirs
from os.path import exists

config  = ConfigParser.ConfigParser()
config.read("par.ini")

queue   = config.get(   'slurm','queue')
N       = 1 # config.getint('slurm','nodes')
n	    = 1 # config.getint('slurm','tasks-per-node')
c       = config.getint('slurm','ncpus')

label   = config.get('mcmc','label')
output  = 'output/'+label

if not exists(output):
	makedirs(output)

content = ('#!/bin/bash\n\n'
		
		   '#SBATCH -p '+queue+'\n'
		   '#SBATCH --nodes='+str(N)+'\n'
		   '#SBATCH --ntasks-per-node='+str(n)+'\n'
		   '#SBATCH --cpus-per-task='+str(c)+'\n'
		   '#SBATCH --exclusive\n'		   
		   '#SBATCH -J mcmc\n' 
		   '#SBATCH -o '+output+'/rsd_mcmc.stdout\n'
		   '#SBATCH -e '+output+'/rsd_mcmc.stderr\n\n'
		   
		   'module unload mpi4py\n\n'
		   
		   'export OMP_NUM_THREADS=1\n'
		   'export NUM_THREADSPROCESSES=${SLURM_CPUS_PER_TASK}\n\n'
		   
		   './main.py')

slurm = open('rsd_mcmc.slurm', 'w')
slurm.write(content)
slurm.close()
