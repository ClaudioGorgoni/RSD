#!/usr/bin/env python
import ConfigParser
import os
import time

from mcmc import p0_init

config = ConfigParser.ConfigParser()
config.read("par.ini")

peak_background = config.getboolean('mcmc', 'peak_background')

label       = config.get(       'mcmc', 'label'      )
pso_init    = config.getboolean('mcmc', 'pso_init'   )
nparticles  = config.getint(    'mcmc', 'Nparticles' )
alpha_p_0   = config.getfloat(  'mcmc', 'alpha_p_0'  )
alpha_v_0   = config.getfloat(  'mcmc', 'alpha_v_0'  )
fz_0        = config.getfloat(  'mcmc', 'fz_0'       )
sigma_FoG_0 = config.getfloat(  'mcmc', 'sigma_FoG_0')
F1_0        = config.getfloat(  'mcmc', 'F1_0'       )
F2_0        = config.getfloat(  'mcmc', 'F2_0'       )

user_init = None
if pso_init:
	if peak_background:
		user_init = [alpha_p_0, alpha_v_0, fz_0, F1_0, sigma_FoG_0]
	else:
		user_init = [alpha_p_0, alpha_v_0, fz_0, F1_0, F2_0, sigma_FoG_0]
#------------------------------------------------------------------------

print "Beginning the PSO optimization..."
start = time.time()

bestfit, fitness = p0_init(nparticles, pso_init, user_init)

total_time = time.time() - start
print "PSO optimization done!\n"

output = 'output/'+label
tmp    = open(output+'/pso_best_fit_info.txt', 'w')

content = ("PSO optimizer result:\n"
           "---------------------\n"
           "Using "+str(nparticles)+" particles\n"
           "Best fit : "+str(bestfit)+"\n"
           "Fitness  : "+str(fitness)+"\n"
           )
content += ("Time to maximize the Likelihood : %.2f sec\n" % total_time)
tmp.write(content)
tmp.close()
