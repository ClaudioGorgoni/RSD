import collections as collec
import sys
from multiprocess import Pool, cpu_count

import GS_sig_pi as gs
import numpy as np
from scipy.integrate import fixed_quad
from scipy.interpolate import splev


# The Legendre polynomials
def P_l(x, l):

	if l == 0:
		return 1.

	elif l == 2:
		return 0.5 * (3. * x * x - 1.)

	elif l == 4:
		return (35. * x ** 4. - 30. * x * x + 3.) / 8.

	else:
		sys.exit("Order of Legendre polynomial must be 0,2 or 4 !")



def xi_sigma_pi(s_par, s_per, fz, sigma_FoG, F1, F2):
	"""Return the GS integral value"""

	f11 = F1 * F2  # F'F''
	f20 = F1 * F1  # F'^2
	f02 = F2 * F2  # F''^2

	# Do the cubic interpolation of xi(r), v12(r), s12_par(r) and s12_perp(r)
	i_xi_r  = gs.interp_xi_r(   F1, F2, f20, f11, f02)
	i_v12   = gs.interp_v12(    F1, F2, f20, f11, f02)
	i_s12_p = gs.interp_s12_par(F1, F2, f20, f11, f02)
	i_s12_v = gs.interp_s12_per(F1, F2, f20, f11, f02)
	
	return gs.integrate_GS(s_par, s_per, i_xi_r, i_v12, i_s12_p, i_s12_v, fz, sigma_FoG)



def xi_smu(mu, s, alpha_p, alpha_v, fz, sigma_FoG, F1, F2):
	"""
	Return the GS integral value, given (mu, s)

	INPUT :
	-------
	mu        : (float) cosine angle between the LOS separation and the pair separation vectors
	s         : (float) scale length in the redshift space
	alpha_p   : (float) the parallel distortion parameter
	alpha_v   : (float) the perpendicular distortion parameter
	fz        : (float) the growth rate
	sigma_FoG : (float) finger of God velocity dispersion
	F1        : (float) 1st order Lagrangian bias
	F2        : (float) 2nd order Lagrangian bias

	OUTPUT :
	--------
	res       : (float) integrand of the GS model
	"""

	s_par = s * mu
	s_per = s * np.sqrt(1. - mu * mu)

	return xi_sigma_pi(alpha_p * s_par, alpha_v * s_per, fz, sigma_FoG, F1, F2)



def inner_integrand(mu, s, l, alpha_p, alpha_v, fz, sigma_FoG, F1, F2):
	"""
	Integrand of the multipole of order l
	---

	P_l(mu) is the Legendre polynomial of order l.
	l must be in [0,2,4]
	"""

	return xi_smu(mu, s, alpha_p, alpha_v, fz, sigma_FoG, F1, F2) * P_l(mu, l) * (2.*l + 1.)
	
	

def vect_inner_integrand(mu, s, l, alpha_p, alpha_v, fz, sigma_FoG, F1, F2):
	
	foo = lambda x : inner_integrand(x, s, l, alpha_p, alpha_v, fz, sigma_FoG, F1, F2)
	return np.array(map(foo, np.array(mu)))



def mu_integral(args):
	"""
	Integrate "inner_integrand" along mu using quad

	INPUT :
	-------

	args  : (tuple) (s, l, alpha_p, alpha_v, fz, sigma_FoG, F1, F2) 
			with s the comoving distance scale and l the multipole order
	"""

	return fixed_quad(vect_inner_integrand, 0., 1.,
					  args=(args[0],args[1],args[2],args[3],args[4],args[5],args[6],args[7]))[0]



def xi_l(s, l, alpha_p, alpha_v, fz, sigma_FoG, F1, F2):
	"""
	Return the multipole prediction of order l for the CLPT+GS model

	"""	
	
	#if isinstance(s, (collec.Sequence, np.ndarray)):

	pool = Pool(cpu_count())

	tmp  = [ (ss, l, alpha_p, alpha_v, fz, sigma_FoG, F1, F2) for ss in s]
		
	res  = np.array( pool.map(mu_integral, tmp) )

	#res = np.array( map(mu_integral, tmp) )
		
	pool.close()
	pool.join()

	#else:

	#	res = mu_integral((s, l, alpha_p, alpha_v, fz, sigma_FoG, F1, F2))

	return res


def xi_02(s, alpha_p, alpha_v, fz, sigma_FoG, F1, F2):
	"""
	Return the l=0,2 multipoles prediction for the CLPT+GS model

	"""	
	
	#if isinstance(s, (collec.Sequence, np.ndarray)):
	
	tmp =  [ (ss, 0, alpha_p, alpha_v, fz, sigma_FoG, F1, F2) for ss in s]
	tmp += [ (ss, 2, alpha_p, alpha_v, fz, sigma_FoG, F1, F2) for ss in s]
		
	pool = Pool(cpu_count())
			
	res  = np.array( pool.map( mu_integral, tmp) )
			
	pool.close()
	pool.join()
	
	#else:
	
	#	tmp  =  [ (s, 0, alpha_p, alpha_v, fz, sigma_FoG, F1, F2) ]
	#	tmp  += [ (s, 2, alpha_p, alpha_v, fz, sigma_FoG, F1, F2) ]
		
	#	pool = Pool()
			
	#	res  = np.array( pool.map( mu_integral, tmp) )
			
	#	pool.close()
	#	pool.join()
		
	return res


def xi_024(s, alpha_p, alpha_v, fz, sigma_FoG, F1, F2):
	"""
	Return the l=0,2,4 multipoles prediction for the CLPT+GS model

	"""

	#if isinstance(s, (collec.Sequence, np.ndarray)):

	tmp =  [(ss, 0, alpha_p, alpha_v, fz, sigma_FoG, F1, F2) for ss in s]
	tmp += [(ss, 2, alpha_p, alpha_v, fz, sigma_FoG, F1, F2) for ss in s]
	tmp += [(ss, 4, alpha_p, alpha_v, fz, sigma_FoG, F1, F2) for ss in s]

	pool = Pool(cpu_count())

	res = np.array( pool.map(mu_integral, tmp) )

	pool.close()
	pool.join()

	#else:

	#	tmp =  [(s, 0, alpha_p, alpha_v, fz, sigma_FoG, F1, F2)]
	#	tmp += [(s, 2, alpha_p, alpha_v, fz, sigma_FoG, F1, F2)]
	#	tmp += [(s, 4, alpha_p, alpha_v, fz, sigma_FoG, F1, F2)]

	#	pool = Pool()

	#	res = np.array( pool.map(mu_integral, tmp) )

	#	pool.close()
	#	pool.join()

	return res
