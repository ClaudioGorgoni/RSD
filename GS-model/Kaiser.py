#!/usr/bin/env python2

import numpy as np
from scipy.integrate import simps
import ConfigParser
from scipy.special import spherical_jn
import sys

# Get power spectrum file from CAMB
Pk_file = "../camb/Pk_lin_z0.57.dat"

kh, Pk = np.loadtxt(Pk_file, unpack=True)

pi = np.pi


# Spherical Fourier transform of a radial function f(k) : SphFT[f(k)] = g(r)
def SphFT(f, x, k, l, a=0.):
	# INPUT
	# -----
	# f : (array) the function to transform from k-space to real space
	# x : (array or float) real values at which to evaluate the Fourier transform
	# k : (array) k-values in Fourier space (must be of the same size of f)
	# a : (float) a real number for better numerical convergence of the integral
	# l	: (int) order of the spherical bessel function --> l = 0, 2, 4
	#
	# OUTPUT
	# ------
	# res (array or float) the result of the spherical Fourier transform


	if l not in [0, 2, 4]:
		sys.exit("Order of the multipole must be 0, 2 or 4 !")

	else:
		res = np.array(
			[simps(k * k * spherical_jn(l, k * ir) * np.exp(-k * k * a * a) * f, k) / (2. * pi * pi) for ir in x])

	return res


# Implementation of the Kaiser formula for RSD in linear regime
def xi_kaiser_l(f, b, r, l):
	# INPUT
	# -----
	# f : (float) the growth rate of structure
	# b : (float) Eulerian bias of galaxies
	# r : (array) the length scale in real space
	# l	: (int) order of the multipole n = 0, 2, 4
	#
	# OUTPUT
	# ------
	# res (array) the multipole of order l in redshift space

	b_eff = 0.

	if l == 0:
		b_eff = b * b + 2. * b * f / 3. + f * f / 5.

	elif l == 2:
		b_eff = -4. * (b * f / 3. + f * f / 7.)

	elif l == 4:
		b_eff = 8. * f * f / 35.

	return b_eff * SphFT(Pk, r, kh, l)
	
	
	
