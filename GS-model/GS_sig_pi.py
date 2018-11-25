import numpy as np
from scipy.integrate import quad
from scipy.interpolate import splrep, splev
import ConfigParser

config = ConfigParser.ConfigParser()
config.read("par.ini")

# Get output files from CLPT
xi_file  = config.get('input', 'xi')
v12_file = config.get('input', 'v12')
s12_file = config.get('input', 's12')

# The scale r
r = np.loadtxt(xi_file, unpack=True)[0]
r = np.insert(r, 0, 0.)  # add 0 to the array
min_r = min(r)
max_r = max(r)

sqrt2pi = np.sqrt( 2. * np.pi )
myeps   = 1.e-12

# THE CLPT OUTPUT
tmp_f1  = np.loadtxt(xi_file, unpack=True)
tmp_xi  = np.array( [tmp_f1[2], tmp_f1[3], tmp_f1[4], tmp_f1[5], tmp_f1[6], tmp_f1[7]] )

tmp_f2  = np.loadtxt(v12_file, unpack=True)
tmp_v12 = np.array( [tmp_f2[2], tmp_f2[3], tmp_f2[4], tmp_f2[5], tmp_f2[6]] )

tmp_f3  = np.loadtxt(s12_file, usecols=(0, 1, 2, 3, 4), unpack=True)
tmp_sp  = np.array( [tmp_f3[1], tmp_f3[2], tmp_f3[3], tmp_f3[4]] )

tmp_f4  = np.loadtxt(s12_file, usecols=(0, 5, 6, 7, 8), unpack=True)
tmp_sv  = np.array( [tmp_f4[1], tmp_f4[2], tmp_f4[3], tmp_f4[4]] )

#-------------------------------------------------------------------

def get_r(): 
	return r


def xi_r(f10, f01, f20, f11, f02):
	"""read xi from file"""
	
	ff  = np.array( [1., f10, f01, f20, f11, f02] )

	val = np.dot(ff, tmp_xi)

	return np.insert(val, 0, val[0] * 1.5)



def v12(f10, f01, f20, f11, f02):
	"""read v12 from file"""
	
	ff  = np.array( [1., f10, f01, f20, f11] )

	val = np.dot(ff, tmp_v12)

	val = np.insert(val, 0, 0.)

	val /= (1. + xi_r(f10, f01, f20, f11, f02))

	return val



def s12_par(f10, f01, f20, f11, f02):
	"""
	read s12 from file
	- parallel component
	"""
	
	ff  = np.array( [1., f10, f01, f20] )

	val = np.dot(ff, tmp_sp)

	val = np.insert(val, 0, val[0])

	val /= (1. + xi_r(f10, f01, f20, f11, f02))

	return val



def s12_per(f10, f01, f20, f11, f02):
	"""
	read s12 from file
	- perpendicular component
	"""
	
	ff  = np.array( [1., f10, f01, f20] )

	val = np.dot(ff, tmp_sv)

	val = np.insert(val, 0, val[0])

	val /= (1. + xi_r(f10, f01, f20, f11, f02))

	return val * 0.5



def interp_xi_r(f10, f01, f20, f11, f02):
	"""Interpolation of xi(r)"""
	return splrep(r, xi_r(f10, f01, f20, f11, f02))



def interp_v12(f10, f01, f20, f11, f02):
	"""Interpolation of v12(r)"""
	return splrep(r, v12(f10, f01, f20, f11, f02))



def interp_s12_par(f10, f01, f20, f11, f02):
	"""Interpolation of s12_par(r)"""
	return splrep(r, s12_par(f10, f01, f20, f11, f02))



def interp_s12_per(f10, f01, f20, f11, f02):
	"""Interpolation of s12_perp(r)"""
	return splrep(r, s12_per(f10, f01, f20, f11, f02))



def inner_integrand(y, s_par, s_per, i_xi_r, i_v12, i_s12_par, i_s12_per, fz, sigma_FoG):
	"""
	The integrand of the GS model

	INPUT :
	-------
	y    	   : (float) Line of sight separation in real space
	s_par	   : (float) Line of sight separation in redshift space
	s_per 	   : (float) perpendicular separation in redshift space
	i_xi_r     : (interpolator) interpolation of xi(r) ; same for v12 and s12
	sigma_FoG  : (float) additional velocity dispersion produced by Finger-of-God effect

	OUTPUT :
	--------
	res        : (float) integrand of the GS model
	"""

	rr  = np.sqrt(s_per * s_per + y * y)

	if rr < min_r or rr > max_r:  # return 0 if rr is outside the range of definition
		res = 0.

	else:

		mu     = y / rr
		mu2    = mu * mu
		
		vinf   = mu*splev(rr, i_v12)*fz

		sigma2 = fz*fz*(splev(rr, i_s12_par)*mu2 + splev(rr, i_s12_per)*(1. - mu2)) + sigma_FoG**2.

		if sigma2 < 0:
			res = 0.

		else:
			exparg = np.exp( - (s_par - y - vinf) * (s_par - y - vinf) / sigma2 / 2. )

			res    = (1. + splev(rr, i_xi_r)) * exparg / np.sqrt(sigma2)

	return res



def integrate_GS(s_par, s_per, i_xi_r, i_v12, i_s12_par, i_s12_per, fz, sigma_FoG):
	"""Integrate function "inner_integrand" along y using quad"""

	res = quad(inner_integrand, s_par - 200., s_par + 200.,
	           args=(s_par, s_per, i_xi_r, i_v12, i_s12_par, i_s12_per, fz, sigma_FoG,),
	           epsabs=1.e-2, epsrel=1.e-2)[0]
	           
	res /= sqrt2pi
	res -= 1.0

	return res

# ---------------END-------------------



