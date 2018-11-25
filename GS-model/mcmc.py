import ConfigParser

import numpy as np
from GS_multi import xi_l, xi_02, xi_024
from cosmoHammer import ParticleSwarmOptimizer
from load_data import load_data
from peak_split import Lagrangian_bias2

config = ConfigParser.ConfigParser()
config.read("par.ini")


N_multi = []
if config.get('input', 'data_xi0') != '':
	N_multi.append(0)
if config.get('input', 'data_xi2') != '':
	N_multi.append(2)
if config.get('input', 'data_xi4') != '':
	N_multi.append(4)
N_multi.sort()

cond0 = 0 in N_multi
cond2 = 2 in N_multi
cond4 = 4 in N_multi

peak_background = config.getboolean('mcmc', 'peak_background')
ndim = 0
if peak_background:
	ndim = 5
else:
	ndim = 6


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
#-------------------------------------------------------

# Extract the data
xi0_file      = config.get('input', 'data_xi0')
xi2_file      = config.get('input', 'data_xi2')
xi4_file      = config.get('input', 'data_xi4')
cov_file      = config.get('input', 'data_cov')
#----------------------------------------------

r_min         = config.getfloat('input', 'r_min')
r_max         = config.getfloat('input', 'r_max')
#------------------------------------------------

# Initialize data
Nmocks	      	       = config.getfloat('input', 'Nmocks')
s, data, icov, log_det = load_data(xi0_file, xi2_file, xi4_file, cov_file, r_min, r_max, Nmocks, ndim)
#-----------------------------------------------------------------------------------------------------
	


def chisquare(diff, icov):
	"""The chisquare from the covariance matrix"""
	return np.dot(diff, np.dot(icov, diff))



def gelman_rubin(chain):

    ssq = np.var(chain, axis=1, ddof=1)
    W   = np.mean(ssq, axis=0)
    Tb  = np.mean(chain, axis=1)
    Tbb = np.mean(Tb, axis=0)
    m   = chain.shape[0]
    n   = chain.shape[1]
    B   = n / (m - 1.) * np.sum((Tbb - Tb)**2., axis=0)
    var = (n - 1.) / n * W + 1. / n * B
    RR  = np.sqrt(var / W)
    return RR
	


def lnprior(theta):
	"""
	Verify that the paramters (theta) respect the priors

	INPUT :
	-------

	theta : (tuple) the input fitting parameters
	"""
	
	if peak_background:
		alpha_p, alpha_v, fz, F1, sigma_FoG     = theta	
		tmp6 = True
	
	else:
		alpha_p, alpha_v, fz, F1, F2, sigma_FoG = theta
		tmp6 = (F2_min     < F2         < F2_max)

	tmp1 = (alpha_p_min    < alpha_p    < alpha_p_max)
	tmp2 = (alpha_v_min    < alpha_v    < alpha_v_max)
	tmp3 = (fz_min         < fz         < fz_max)
	tmp4 = (sigma_FoG_min  < sigma_FoG  < sigma_FoG_max)
	tmp5 = (F1_min         < F1         < F1_max) 

	if tmp1 and tmp2 and tmp3 and tmp4 and tmp5 and tmp6:
		return 0.0

	return -np.inf



def lnlike(theta):
	"""
	Compute the log-likelihood as -chi^2/2

	INPUT :
	-------

	theta   : (tuple) the input fitting parameters
	s       : (array) the distance scale array
	data    : (array) the observed multipoles of the 2pcf
	icov    : (2d-array) the inverse covariance matrix of the data
	log_det : (float) the log of the determinant of the covariance matrix

	OUTPUT :
	-------

	res     : (float) the log-likelihood
	"""
	
	if peak_background:
		alpha_p, alpha_v, fz, F1, sigma_FoG     = theta
		F2 = Lagrangian_bias2(F1)
	
	else:
		alpha_p, alpha_v, fz, F1, F2, sigma_FoG = theta
		
	
	if cond0 and cond2 and not cond4:
		multi_model = xi_02(s, alpha_p, alpha_v, fz, sigma_FoG, F1, F2)
	
	elif cond0 and cond2 and cond4:
		multi_model = xi_024(s, alpha_p, alpha_v, fz, sigma_FoG, F1, F2)
	
	else:
		multi_model = np.hstack([ xi_l(s, ll, alpha_p, alpha_v, fz, sigma_FoG, F1, F2) for ll in N_multi ])

	tmp = data - multi_model
	
	return - 0.5 * np.dot(tmp, np.dot(icov, tmp))


def lnprob(theta):
	"""Compute the posterior distribution"""

	lp = lnprior(theta)

	if not np.isfinite(lp):
		return -np.inf

	return lp + lnlike(theta)
		
		
		
def p0_init(nparticles, pso_init, guess):
	"""
	Maximize the Likelihood to start from an optimal position for the MCMC chain

	INPUT :
	-------

	nparticles : (int) The number of particles to use in the PSO.
				       All the other parameters are set to default values.

	OUTPUT :
	--------

	theta_init : (tuple) The best fit position
	fitness    : (float) The fitness of the best fit
	"""
	
	if peak_background:
		bnds_min = [alpha_p_min, alpha_v_min, fz_min, F1_min, sigma_FoG_min]
		bnds_max = [alpha_p_max, alpha_v_max, fz_max, F1_max, sigma_FoG_max]
		
	else:
		bnds_min = [alpha_p_min, alpha_v_min, fz_min, F1_min, F2_min, sigma_FoG_min]
		bnds_max = [alpha_p_max, alpha_v_max, fz_max, F1_max, F2_max, sigma_FoG_max]
	

	pso = ParticleSwarmOptimizer(lnprob, low=bnds_min, high=bnds_max, particleCount=nparticles)
	
	if pso_init and guess is not None:
		pso.gbest.position = guess
		pso.gbest.velocity = [0]*len(guess)
		pso.gbest.fitness  = lnprob(guess)
	
	# sample the parameter space to find the max Likelihood
	for swarm in pso.sample(maxIter=500):
		continue
	
	return pso.gbest.position, pso.gbest.fitness
		
