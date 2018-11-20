import numpy as np
import ConfigParser
from astropy.cosmology import wCDM
import astropy.units as u
from scipy.integrate import quad


config = ConfigParser.ConfigParser()
config.read("par.ini")


h      = config.getfloat('cosmo', 'h'     )
ombh2  = config.getfloat('cosmo', 'ombh2' )
omm    = config.getfloat('cosmo', 'omm'   )
omL    = config.getfloat('cosmo', 'omL'   )
sigma8 = config.getfloat('cosmo', 'sigma8')
ns     = config.getfloat('cosmo', 'ns'    )
mnu    = config.getfloat('cosmo', 'mnu'   )
omk    = config.getfloat('cosmo', 'omk'   )
w      = config.getfloat('cosmo', 'w'     )
	
cosmo  = wCDM(H0=h*100., Om0=omm, Ode0=omL, w0=w, Tcmb0=2.7255, Ob0=ombh2/h/h, m_nu = mnu * u.eV) 

c      = 299792.458 # km/s
z_dec  = 1059.68    # the redshift of the baryon drag epoch


# The Hubble parameter in km/s/Mpc
def Hubble(z):
	return cosmo.H(z).value


# The angular diameter distance in Mpc
def Angular_dist(z):
	return cosmo.angular_diameter_distance(z).value
	

# The speed of sound inside the coupled baryon-photon plasma
def v(z):
	return c / np.sqrt(3.*(1. + 3.*cosmo.Ob0/(4.*cosmo.Ogamma0*(1.+z))))


# The integrand of the BAO scale
def integral(z):
	return v(z) / Hubble(z)


# The sound horizon
def rs_fid():
	return quad(integral, z_dec, np.inf)[0]
	
	
# g(z) = H(z)/H0
def g(z):
	return cosmo.efunc(z)


# The total matter density given at redshift z
def OmegaM(z):
	return cosmo.Om(z)
	

# The growth function
def Growth_func(z):
	
	func = lambda x: (1. + x) / g(x)**3.

	return 2.5 * omm * g(z) * quad(func, z, np.inf)[0]
	

# The exact solution of the growth rate for a LCDM universe computed from f = dlnD(a)/dln(a)
def Growth_rate(z_):
	return 0.5 * omm * (1. + z_) * (1. + z_) * (5. / Growth_func(z_) - 3. * (1. + z_) - omk / omm) / g(z_) / g(z_)


def Growth_rate_approx(z_):
	return OmegaM(z_) ** 0.545


# The amplitude of the matter power spectrum at 8 Mpc/h
def sigma8_z(z):
	return sigma8 * Growth_func(z)/Growth_func(0.)

