import ConfigParser
import os

import matplotlib
import numpy as np
from cosmo_lib import sigma8_z, Hubble, Angular_dist, rs_fid
from mcmc import lnlike

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
from os.path import exists
from os import makedirs
from load_data import load_data

import warnings
warnings.filterwarnings("ignore")

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)


config = ConfigParser.ConfigParser()
config.read("par.ini")


label  = config.get('mcmc','label')
output = 'output/'+label
if not exists(output):
	makedirs(output)

peak_background = config.getboolean('mcmc', 'peak_background')

# Setup the labels
triangle_labels = ["\\alpha_{\parallel}", "\\alpha_{\perp}", "f\sigma_8(z)", "b\sigma_8(z)"]
		  	  	   
hubble_labels   = ["H(z)r_s(z_d)", "D_A(z)/r_s(z_d)", "f\sigma_8(z)"]

labels          = [r'$\alpha_{\parallel}$', r'$\alpha_{\perp}$', r'$f(z)$', r'$F_1$']
#-------------------------------------------------------------------------------------------


ndim = 0
if peak_background:
	ndim = 5	
else:
	ndim = 6
	labels.append(r'$F_2$')  
labels.append(r'$\sigma_{FoG}$')
#-------------------------------


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
Ndata = len( load_data(xi0_file, xi2_file, xi4_file, cov_file, r_min, r_max, Nmocks, ndim)[1] )
#-----------------------------------------------------------------------------------------------------
	



	
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
    
    
    
def write_chains(chains):
	
	for i in range(ndim):
		
		tmp = chains[:,:,i].T
		np.savetxt(output+'/chain_'+str(i+1)+'.dat', tmp)
#---------------------------------------------------------------------------------
		  	  


def plot_chains(chains):
	
	for i in range(ndim):
		
		tmp = chains[:,:,i].T
		
		plt.figure()
		plt.plot(tmp, '-', color='k', alpha=0.3)
		plt.xlabel(r'step number', size=16)
		plt.ylabel(labels[i], size=16)
		plt.savefig(output+'/chain_'+str(i+1)+'.png', bbox_inches='tight', format='png')
#-----------------------------------------------------------------------------------------------------



### Write information file about all the mcmc parameters
def write_mcmc_results(samples, chains):
	
	tmp     = open(output+'/mcmc_info1.txt', 'w')
	content = ""
	
	for i in range(ndim):
		content += ("Gelman-Rubin for chain "+str(i+1)+": R-1 = %.3f\n" % (gelman_rubin(chains[:,:,i]) - 1.) )
	
	content += "\n"
	
	if not peak_background:
		ap_mcmc, av_mcmc, f_mcmc, f1_mcmc, f2_mcmc, sig_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
		zip(*np.percentile(samples, [16, 50, 84], axis=0)))
		
		chi2 = -2.*lnlike((ap_mcmc[0], av_mcmc[0], f_mcmc[0], f1_mcmc[0], f2_mcmc[0], sig_mcmc[0]))
		ddof = Ndata - 1 - 6
		
		content += ("chi2/ddof  = %.1f/%d\n\n" % (chi2, ddof))
		
		content += ("alpha_para = %.3f (+%.3f / -%.3f)\n" % ( ap_mcmc[0],  ap_mcmc[1],  ap_mcmc[2]))
		content += ("alpha_perp = %.3f (+%.3f / -%.3f)\n" % ( av_mcmc[0],  av_mcmc[1],  av_mcmc[2]))
		content += ("f(z)       = %.3f (+%.3f / -%.3f)\n" % (  f_mcmc[0],   f_mcmc[1],   f_mcmc[2]))
		content += ("F1         = %.3f (+%.3f / -%.3f)\n" % ( f1_mcmc[0],  f1_mcmc[1],  f1_mcmc[2]))
		content += ("F2         = %.3f (+%.3f / -%.3f)\n" % ( f2_mcmc[0],  f2_mcmc[1],  f2_mcmc[2]))
		content += ("sigma_fog  = %.3f (+%.3f / -%.3f)\n" % (sig_mcmc[0], sig_mcmc[1], sig_mcmc[2]))
		
	else:
		ap_mcmc, av_mcmc, f_mcmc, f1_mcmc, sig_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
		zip(*np.percentile(samples, [16, 50, 84], axis=0)))
		
		chi2 = -2.*lnlike((ap_mcmc[0], av_mcmc[0], f_mcmc[0], f1_mcmc[0], sig_mcmc[0]))
		ddof = Ndata - 1 - 5
		
		content += ("chi2/ddof  = %.1f/%d\n\n" % (chi2, ddof))
		
		content += ("alpha_para = %.3f (+%.3f / -%.3f)\n" % ( ap_mcmc[0],  ap_mcmc[1],  ap_mcmc[2]))
		content += ("alpha_perp = %.3f (+%.3f / -%.3f)\n" % ( av_mcmc[0],  av_mcmc[1],  av_mcmc[2]))
		content += ("f(z)       = %.3f (+%.3f / -%.3f)\n" % (  f_mcmc[0],   f_mcmc[1],   f_mcmc[2]))
		content += ("F1         = %.3f (+%.3f / -%.3f)\n" % ( f1_mcmc[0],  f1_mcmc[1],  f1_mcmc[2]))
		content += ("sigma_fog  = %.3f (+%.3f / -%.3f)\n" % (sig_mcmc[0], sig_mcmc[1], sig_mcmc[2]))
		                        						   
	tmp.write(content)
	tmp.close()
#-------------------------------------------------------------------------------------------------------------



def write_science_results(samples, z):
	
	# do not read sigma_fog
	samples = samples[:,0:-1]
	
	# Multiply the growth rate and the bias by sigma8
	sigma8z      = sigma8_z(z)
	
	# fsigma8
	samples[:,2] = samples[:,2] * sigma8z
	
	# bsigma8
	samples[:,3] = (samples[:,3] + 1.) * sigma8z
	
	if not peak_background:
		samples = samples[:,0:-1]

	
	# write information file about the science parameters
	tmp     = open(output+'/mcmc_info2.txt', 'w')
	content = ""

	ap_mcmc, av_mcmc, fs8_mcmc, bs8_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
	zip(*np.percentile(samples, [16, 50, 84], axis=0)))
		
	content += ("alpha_para = %.3f (+%.3f / -%.3f)\n" % ( ap_mcmc[0],  ap_mcmc[1],  ap_mcmc[2]))
	content += ("alpha_perp = %.3f (+%.3f / -%.3f)\n" % ( av_mcmc[0],  av_mcmc[1],  av_mcmc[2]))
	content += ("fs8(z)     = %.3f (+%.3f / -%.3f)\n" % (fs8_mcmc[0], fs8_mcmc[1], fs8_mcmc[2]))
	content += ("bs8(z)     = %.3f (+%.3f / -%.3f)\n" % (bs8_mcmc[0], bs8_mcmc[1], bs8_mcmc[2]))
		                        						   
	tmp.write(content)
	tmp.close()
	
	
	# triangular plot
	tmp = MCSamples(samples=samples, labels=triangle_labels)
	
	g = plots.getSubplotPlotter()
	g.triangle_plot(tmp, filled=True)
	g.export(output+'/RSD_triangle_plot.pdf')
	g.export(output+'/RSD_triangle_plot.png')
	#---------------------------------------------------------------------------------



	# Plot results for H(z) and D_A(z)
	rs = rs_fid()
	
	# H(z) * r_s(z_d) / 1000
	samples[:,0] = Hubble(z) * rs / samples[:,0] / 1000.
	# D_A(z) / r_s(z_d)
	samples[:,1] = Angular_dist(z) * samples[:,1] / rs
	
	
	# write information file about the science parameters
	tmp     = open(output+'/mcmc_info3.txt', 'w')
	content = ""

	ap_mcmc, av_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
	zip(*np.percentile(samples[:,0:2], [16, 50, 84], axis=0)))
	
	content += ("The fiducial value for the sound horizon rs_fid(zd) = %.3f Mpc\n\n" % rs)
	content += ("H(z)rs(zd)    = %.3f 10^3 (+%.3f / -%.3f)\n" % ( ap_mcmc[0],  ap_mcmc[1],  ap_mcmc[2]))
	content += ("D_A(z)/rs(zd) = %.3f (+%.3f / -%.3f)\n" % ( av_mcmc[0],  av_mcmc[1],  av_mcmc[2]))
	content += ("fs8(z)        = %.3f (+%.3f / -%.3f)\n" % (fs8_mcmc[0], fs8_mcmc[1], fs8_mcmc[2]))
		                        						   
	tmp.write(content)
	tmp.close()
	
	
	# triangular plot
	tmp = MCSamples(samples=samples[:,0:3], labels=hubble_labels)
	
	g = plots.getSubplotPlotter()
	g.triangle_plot(tmp, filled=True)
	g.export(output+'/RSD_hubble_plot.pdf')
	g.export(output+'/RSD_hubble_plot.png')
#---------------------------------------------------------------------------------------




