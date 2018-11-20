#!/usr/bin/env python
import ConfigParser
import os
import time

from mcmc import p0_init

config = ConfigParser.ConfigParser()
config.read("par.ini")

nparticles = config.getint('mcmc', 'Nparticles')

print "Beginning the PSO optimization..."
start = time.time()

bestfit, fitness = p0_init(nparticles)

total_time = time.time() - start
print "PSO optimization done!\n"

tmp = open(output+'/pso_best_fit_info.txt', 'w')

content = ("PSO optimizer result:\n"
           "---------------------\n"
           "Using "+str(nparticles)+" particles\n"
           "Best fit : "+str(bestfit)+"\n"
           "Fitness  : "+str(fitness)+"\n"
           )
content += ("Time to maximize the Likelihood : %.2f sec\n" % total_time)
tmp.write(content)
tmp.close()