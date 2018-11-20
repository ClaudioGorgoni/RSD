import ConfigParser
import cPickle as pickle

from mcmc_plot_lib import write_chains, plot_chains, write_mcmc_results, write_science_results

config = ConfigParser.ConfigParser()
config.read("par.ini")

label    = config.get('mcmc','label')
z_sample = config.getfloat('cosmo','z')


# Get the samples and the chains
output  = 'output/'+label
samples = pickle.load( open( output+"/samples.s", "r" ) )
chains  = pickle.load( open( output+"/chains.c",  "r" ) )

# Plot the chains at each step
write_chains(chains)
print "chains saved!"

# Plot the chains at each step
plot_chains(chains)
print "chains plotted!"

# Write informations about the mcmc parameters
write_mcmc_results(samples, chains)
print "MCMC results saved!"

# Write science results in files and plots
write_science_results(samples, z_sample)
print "Nice elliptical plots created!"
print "ALL DONE, GOOD BYE!!!...\n"