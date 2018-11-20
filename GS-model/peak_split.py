import numpy as np 
from scipy.interpolate import splev, splrep



def peak_background_bias(nu):
	"""
	INPUT :
	-------
	nu : (float) Input value for peak background split.
		         This is helpful if we want to make our basis set f, nu and sigma_FoG.

	OUTPUT:
	-------
	F1 : (float) First order Lagrangian bias parameter

    F2 : (flaot) Second order Lagrangian bias parameter
	"""

	delc = 1.686
	a    = 0.707
	p    = 0.3
	anu2 = a * nu * nu

	F1 = (anu2 - 1. + 2. * p / (1. + anu2 ** p)) / delc
	F2 = (anu2**2. - 3. * anu2 + 2. * p * (2. * anu2 + 2. * p - 1.) / (1. + anu2 ** p)) / delc / delc

	return F1, F2
	
	
# ---- Invert the function ----
nu     = np.linspace(0.,5.,10000)
f1, f2 = peak_background_bias(nu)

tmp    = (f1 >= -0.2)

tck    = splrep(f1[tmp], nu[tmp])



def nu_split(lag_bias):
	"""
	INPUT :
	-------
	lag_bias : (float) The 1st order Lagrangian bias

	OUTPUT:
	-------
	nu 		 : (float) the peak backgrouns split parameter
	"""
	
	if lag_bias < -0.2:
		sys.exit("Lagrangian bias must be greater than -0.2")
	
	return splev(lag_bias, tck)
	


def Lagrangian_bias2(lag_bias):
	"""
	INPUT :
	-------
	lag_bias : (float) The 1st order Lagrangian bias

	OUTPUT:
	-------
    F2 : (flaot) 2nd order Lagrangian bias parameter
	"""
	
	return peak_background_bias( nu_split(lag_bias) )[1]
	

	
