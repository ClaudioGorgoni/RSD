#!/usr/bin/env python2

import numpy as np
import emcee
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import time
from getdist import plots, MCSamples

plt.rc('font',  family='serif')
plt.rc('text',  usetex=True)
plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16)

def gelman_rubin(chain):
    ssq = np.var(chain, axis=1, ddof=1)
    W   = np.mean(ssq, axis=0)
    Tb  = np.mean(chain, axis=1)
    Tbb = np.mean(Tb, axis=0)
    m   = chain.shape[0] * 1.0
    n   = chain.shape[1] * 1.0
    B   = n / (m - 1.) * np.sum((Tbb - Tb)**2., axis=0)
    var = (n - 1.) / n * W + 1. / n * B
    RR  = np.sqrt(var / W)
    return RR

# Choose the "true" parameters.
m_true = -0.9594
b_true = 4.294
f_true = 0.534

# Generate some synthetic data from the model.
N = 50
x = np.sort(10*np.random.rand(N))
yerr = 0.1+0.5*np.random.rand(N)
y = m_true*x+b_true
y += np.abs(f_true*y) * np.random.randn(N)
y += yerr * np.random.randn(N)

def lnlike(theta, x, y, yerr):
    m, b, lnf = theta
    model = m * x + b
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

import scipy.optimize as op
nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, [m_true, b_true, np.log(f_true)], args=(x, y, yerr))
#m_ml, b_ml, lnf_ml = result["x"]

def lnprior(theta):
    m, b, lnf = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

#-----------------------------------------------------------------


ndim, nwalkers = 3, 16
p0 = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]


sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))

pos,prob,state = sampler.run_mcmc(p0,100)
sampler.reset()
sampler.run_mcmc(pos, 1000)

labels = [r'$m$', r'$b$', r'$\ln(f)$']

# Burn the first 'mcmc_burn' steps of the walkers
samples = sampler.chain[:, :, :].reshape((-1, ndim))
chains  = sampler.chain[:, :, :]

for k in range(ndim):
	print gelman_rubin(chains[:,:,k]) - 1.
	
label = 'test'



for i in range(ndim):
	
	tmp = sampler.chain[:,:,i].T
	np.savetxt('chain_'+str(i+1)+'-'+str(label)+'.dat', tmp)
#---------------------------------------------------------------------------------


for i in range(ndim):
	
	tmp = sampler.chain[:,:,i].T
	
	plt.figure()
	plt.plot(tmp, '-', color='k', alpha=0.3)
	plt.xlabel(r'step number', size=16)
	plt.ylabel(labels[i], size=16)
	plt.savefig('chain_'+str(i+1)+'-'+str(label)+'.png', bbox_inches='tight', format='png')



samples[:, 2] = np.exp(samples[:, 2])
m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))

content  = ("m = %.3f (+%.3f / -%.3f)\n" % ( m_mcmc[0],  m_mcmc[1],  m_mcmc[2]))
content += ("b = %.3f (+%.3f / -%.3f)\n" % ( b_mcmc[0],  b_mcmc[1],  b_mcmc[2]))
content += ("f = %.3f (+%.3f / -%.3f)\n" % ( f_mcmc[0],  f_mcmc[1],  f_mcmc[2]))

print content
#mcmc_res.write(content)
#mcmc_res.close()

labels = ["m", "b", "f"]

sample1 = MCSamples(samples = samples, labels=labels)
g = plots.getSubplotPlotter()
g.triangle_plot(sample1, filled=True)
g.export('triangle_plot1.pdf')
g.export('triangle_plot1.png')

#plt.show()
