#!/usr/bin/env python
import ConfigParser
import cPickle as pickle
import os
import sys
import time
from multiprocessing import cpu_count
from os import makedirs
from os.path import exists

import emcee
import numpy as np
from mcmc import p0_init, lnprob

config = ConfigParser.ConfigParser()
config.read("par.ini")

peak_background = config.getboolean('mcmc', 'peak_background')

# Get priors for fitting parameters
alpha_p_min   = config.getfloat('mcmc', 'alpha_p_min'  )
alpha_v_min   = config.getfloat('mcmc', 'alpha_v_min'  )
fz_min        = config.getfloat('mcmc', 'fz_min'       )
sigma_FoG_min = config.getfloat('mcmc', 'sigma_FoG_min')
F1_min        = config.getfloat('mcmc', 'F1_min'       )
F2_min        = config.getfloat('mcmc', 'F2_min'       )

alpha_p_max   = config.getfloat('mcmc', 'alpha_p_max'  )
alpha_v_max   = config.getfloat('mcmc', 'alpha_v_max'  )
fz_max        = config.getfloat('mcmc', 'fz_max'       )
sigma_FoG_max = config.getfloat('mcmc', 'sigma_FoG_max')
F1_max        = config.getfloat('mcmc', 'F1_max'       )
F2_max        = config.getfloat('mcmc', 'F2_max'       )
#---------------------------------------------------------


ndim = 0
if peak_background:
	ndim = 5
	bnds_min = [alpha_p_min, alpha_v_min, fz_min, F1_min, sigma_FoG_min]
	bnds_max = [alpha_p_max, alpha_v_max, fz_max, F1_max, sigma_FoG_max]
	
else:
	ndim = 6
	bnds_min = [alpha_p_min, alpha_v_min, fz_min, F1_min, F2_min, sigma_FoG_min]
	bnds_max = [alpha_p_max, alpha_v_max, fz_max, F1_max, F2_max, sigma_FoG_max]
#-------------------------------------------------------------------------------


# some init for multiprocessing
if os.environ.has_key('NUM_THREADSPROCESSES'):
	ncpu = os.environ['NUM_THREADSPROCESSES']
	ncpu = int(ncpu)
else:
	ncpu = cpu_count()

print "Running on %d cores\n" % ncpu


def get_samples(sampler_):
	samples_ = sampler_.chain[:, :, :].reshape((-1, ndim))
	chains_ = sampler_.chain[:, :, :]

	return samples_, chains_



def init_sampler(nparticles_, pso_init_, nwalkers_, guess_ = None):

	if pso_init_:

		print "Finding optimal starting position for MCMC with PSO..."
		start = time.time()

		tmp = p0_init( nparticles_, pso_init_ , guess_ )[0]

		print "Time to maximize the Likelihood : %.2f sec\n" % (time.time() - start)

	else:
		tmp = guess_

	print "Initial position of the MCMC sampling:"
	print tmp

	p0 = [tmp + 0.005 * np.random.randn(len(bnds_min)) for i in range(nwalkers_)]

	for i in range(nwalkers_):
		p0[i] = np.clip(p0[i], bnds_min, bnds_max)

	return p0




# THE MAIN FUNCTION
def run_mcmc(p0_, mcmc_steps_, nwalkers_, nburnin_):


	# Initialize the mcmc sampler
	print "MCMC sampler launched...\n"
	start = time.time()

	sampler = emcee.EnsembleSampler(nwalkers_, ndim, lnprob)
	
	print "Begin the burnin step..."
	pos,prob,state = sampler.run_mcmc(p0_, nburnin_)

	print "Time for the burn-in step : %.2f sec\n" % (time.time() - start)

	# Reset the sampler after the burn-in
	sampler.reset()
	
	print "Begin the mcmc sampling..."
	start = time.time()

	# Launch the MCMC sampling
	sampler.run_mcmc(pos, mcmc_steps_)
	
	print "Total time of the MCMC sampling : %.2f sec" % (time.time() - start)
		
	return sampler
### END OF RUN_MCMC ###




# Load the MCMC conditions for the run
label       = config.get(   'mcmc','label'         )
mcmc_steps  = config.getint('mcmc','Nsteps'        )
mcmc_burn   = config.getint('mcmc','Nburns'        )
nwalkers    = config.getint('mcmc','Nwalkers'      )
#---------------------------------------------------


# Load the user's initial guess
my_guess    = config.getboolean('mcmc', 'my_guess'   )
pso_init    = config.getboolean('mcmc', 'pso_init'   )
nparticles  = config.getint(    'mcmc', 'Nparticles' )
alpha_p_0   = config.getfloat(  'mcmc', 'alpha_p_0'  )
alpha_v_0   = config.getfloat(  'mcmc', 'alpha_v_0'  )
fz_0        = config.getfloat(  'mcmc', 'fz_0'       )
sigma_FoG_0 = config.getfloat(  'mcmc', 'sigma_FoG_0')
F1_0        = config.getfloat(  'mcmc', 'F1_0'       )
F2_0        = config.getfloat(  'mcmc', 'F2_0'       )
#-----------------------------------------------------


# The initial guess of the user (optional)
user_init = None
if my_guess or pso_init:
	if peak_background:
		user_init = [alpha_p_0, alpha_v_0, fz_0, F1_0, sigma_FoG_0]
	else:
		user_init = [alpha_p_0, alpha_v_0, fz_0, F1_0, F2_0, sigma_FoG_0]
#------------------------------------------------------------------------



#################################
### EXECUTE THE MAIN FUNCTION ###
#################################

p0      = init_sampler(nparticles, pso_init, nwalkers, guess_ = user_init)

sampler = run_mcmc(p0, mcmc_steps, nwalkers, mcmc_burn)

samples, chains = get_samples(sampler)

# Save the samples and chains
output = 'output/'+label
if not exists(output):
	makedirs(output)
pickle.dump( samples, open( output+"/samples.s", "w" ) )
pickle.dump( chains,  open( output+"/chains.c",  "w" ) )
print "MCMC SAMPLING DONE!!!\n"

####### END OF MAIN #######
