import numpy as np
import sys


def load_data(f_xi0, f_xi2, f_xi4, f_cov, r_min_, r_max_, Nmocks_, Nparam_):

	N_multi = 0
	
	# The multipoles
	if f_xi0 != '':
		s, xi0 = np.loadtxt(f_xi0, unpack=True, usecols=(0,1))
	
		_r_min_ = s[s >= r_min_ ][0]
		_r_max_ = s[s <= r_max_ ][-1]
	
		i_r_min = np.argwhere(s == _r_min_)[0][0]
		i_r_max = np.argwhere(s == _r_max_)[0][0]

		tmp   = (s >= _r_min_) & (s <= _r_max_)
		xi0   = xi0[tmp]
		
		N_multi += 1
	else:
		sys.exit('Error: no data file for the monopole !')



	if f_xi2 != '':
		xi2 = np.loadtxt(f_xi2, unpack=True, usecols=(1,))
	
		xi2 = xi2[tmp]
		
		N_multi +=1
	else:
		xi2 = np.zeros(0)



	if f_xi4 != '':
		xi4 = np.loadtxt(f_xi4, unpack=True, usecols=(1,))
	
		xi4 = xi4[tmp]
		
		N_multi += 1	
	else:
		xi4 = np.zeros(0)
	
	s    = s[tmp]
	data = np.hstack((xi0,xi2,xi4))
	
	Ndata = len(data)
	#--------------------------------------------------------------
	
	
	
	# The inverse covariance matrix
	if f_cov != '':
		cov = np.loadtxt(f_cov)
	else:
		sys.exit('Error: no data file for the covariance matrix !')
	
	
	
	#---------------------------------------------------------------------
	#N = len(cov)/N_multi
	#
	#
	## Select each covariance matrices 
	#
	#cov_list = []
	#for i in range(N_multi):
	#	for j in range(N_multi):
	#	
	#		cov_list.append( cov[ i*N : i*N + N, j*N : j*N + N] )
	#
	#
	## Get the inverse matrices ad the log of their determinant
	#
	#icov_list = []
	#det_list  = []
	#for i in range(len(cov_list)):
	#
	#	tmp      = cov_list[i][i_r_min: i_r_max + 1, i_r_min: i_r_max + 1]	
	#
	#	det_list.append( np.linalg.slogdet(tmp)[1] )
	#	icov_list.append( np.linalg.inv(tmp) )
	#	
	#
	#log_det = 0.
	#for i in det_list:
	#	log_det += i
	#
	#
	## Assemble the inverse matrix by blocks
	#
	#tmp = []
	#for i in range(N_multi):
	#	tmp.append( np.hstack(icov_list[i*N_multi:(i+1)*N_multi]) )
	#
	#icov = np.vstack(tmp)
	#----------------------------------------------------------------------
	
	
	
	N = int(len(cov)/N_multi)
	
	# Select each covariance matrices 
	cov_list = []
	for i in range(N_multi):
		for j in range(N_multi):
		
			cov_list.append( cov[ i*N : i*N + N, j*N : j*N + N] )
	
	
	# Apply the range cut
	for i in range(len(cov_list)):
	
		cov_list[i] = cov_list[i][i_r_min: i_r_max + 1, i_r_min: i_r_max + 1]
	
	
	# Assemble the matrix by blocks
	tmp = []
	for i in range(N_multi):
		tmp.append( np.hstack(cov_list[i*N_multi : (i+1)*N_multi]) )
	
	cov     = np.vstack(tmp)
	icov    = np.linalg.inv(cov)
	log_det = np.linalg.slogdet(cov)[1]
	
	
	# The unbiased estimator (Percival et al. 2013)
	icov *= ( 1. - (Ndata + 1.) / (Nmocks_ - 1.) )
	
	# Account for uncertainties in the covariance matrix
	A = 2. / (Nmocks_ - Ndata - 1.) / (Nmocks_ - Ndata - 4.)
	B = (Nmocks_ - Ndata - 2.) / (Nmocks_ - Ndata - 1.) / (Nmocks_ - Ndata - 4.)
	
	icov *= (1. + B*(Ndata - Nparam_)) / (1. + A + B*(Nparam_ - 1.))
	
	
	
	if len(icov) != len(data):
		sys.exit('Sizes of data array and covariance matrix do not match !')
	
	#print("The comoving scale array:")
	#print(s)
	#print("The sizes of the scale/data/icov arrays: %.0f / %.0f / (%.0f,%.0f)\n" \
	#% (len(s), len(data), np.shape(icov)[0], np.shape(icov)[1]))

	# chi2_data = np.log(2.*np.pi) + log_det

	return s, data, icov, log_det

#-------------------------------------------------------------

