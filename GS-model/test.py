#!/usr/bin/env python

import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from GS_multi import xi_l, xi_02

r = np.linspace(10,150,16)

start = time.time()
xi0 = xi_l(r, 0, 0.98, 1.0, 0.74, 3.5, 0.95, -0.1)
xi2 = xi_l(r, 2, 0.98, 1.0, 0.74, 3.5, 0.95, -0.1)
print xi0, xi2
print time.time() - start


sys.exit()

s, xi0d = np.loadtxt('../DR12/Cuesta_2016_CMASSDR12_corrfunction_x0_prerecon.dat', usecols=(0,1,), unpack=True)
xi2d    = np.loadtxt('../DR12/Cuesta_2016_CMASSDR12_corrfunction_x2_prerecon.dat', usecols=(1,),   unpack=True)

cov = np.loadtxt('../DR12/Cuesta_2016_CMASSDR12_corrfunction_cov_x0x2_prerecon.dat')
err0 = np.sqrt(np.diag(cov[0:len(s),0:len(s)]))
err2 = np.sqrt(np.diag(cov[len(s):2*len(s),len(s):2*len(s)]))


plt.figure()
plt.plot(r,xi0*r*r,'r-',label='model')
plt.errorbar(s,xi0d*s*s,yerr=err0*s*s,fmt='bo',label='EZ mocks')
plt.xlabel(r'$r$')
plt.ylabel(r'$r^2\xi_0(r)$')
plt.xlim(10,150)
#plt.ylim(0,50)
plt.legend()
plt.grid()

plt.figure()
plt.plot(r,xi2*r*r,'r-',label='model')
plt.errorbar(s,xi2d*s*s,yerr=err2*s*s,fmt='bo',label='EZ mocks')
plt.xlabel(r'$r$')
plt.ylabel(r'$r^2\xi_2(r)$')
plt.xlim(10,150)
#plt.ylim(-70,0)
plt.legend()
plt.grid()

plt.show()



#s, xi0_d = np.loadtxt('../Correlation/output/SGC/58409.58576/ELG_v3_xi0.dat', usecols=(0,1), unpack=True)
#xi2_d    = np.loadtxt('../Correlation/output/SGC/58409.58576/ELG_v3_xi2.dat', usecols=(1),   unpack=True)

#cov  = np.loadtxt('../Covariance/EZ_mocks/SGC_cov_00.dat')
#err0 = np.sqrt(np.diag(cov))
#cov  = np.loadtxt('../Covariance/EZ_mocks/SGC_cov_22.dat')
#err2 = np.sqrt(np.diag(cov))

#rm, mm0, ms0 = np.loadtxt('../Covariance/EZ_mocks/SGC_mean_xi0.dat', usecols=(0,1,2), unpack=True)
#rm, mm2, ms2 = np.loadtxt('../Covariance/EZ_mocks/SGC_mean_xi2.dat', usecols=(0,1,2), unpack=True)

#rmq, mmq0, msq0 = np.loadtxt('../Covariance/qpm_mocks/SGC_mean_xi0.dat', usecols=(0,1,2), unpack=True)
#rmq, mmq2, msq2 = np.loadtxt('../Covariance/qpm_mocks/SGC_mean_xi2.dat', usecols=(0,1,2), unpack=True)

#plt.figure()
#plt.plot(r-1,xi0*r*r,'r-',label='fitted model')
#plt.plot(r-1,xi0_t*r*r,'g-',label='fiducial model')
#plt.errorbar(s,xi0_d*s*s,yerr=err0*s*s,fmt='bo',label='ELG data v3')
#plt.errorbar(rm,mm0*rm*rm,yerr=ms0*rm*rm,fmt='kx',label='EZ mocks')
#plt.errorbar(rmq+1,mmq0*rmq*rmq,yerr=msq0*rmq*rmq,fmt='m.',label='qpm mocks')
#plt.xlabel(r'$r$')
#plt.ylabel(r'$r^2\xi_0(r)$')
#plt.legend()
#plt.grid()
#plt.xlim(20,150)
#plt.ylim(-30,50)

#plt.figure()
#plt.plot(r-1,xi2*r,'r-',label='fitted model')
#plt.plot(r-1,xi2_t*r,'g-',label='fiducial model')
#plt.errorbar(s,xi2_d*s,yerr=err2*s,fmt='bo',label='ELG data v3')
#plt.errorbar(rm,mm2*rm,yerr=ms0*rm,fmt='kx',label='EZ mocks')
#plt.errorbar(rmq+1,mmq2*rmq,yerr=msq2*rmq,fmt='m.',label='qpm mocks')
#plt.xlabel(r'$r$')
#plt.ylabel(r'$r\xi_2(r)$')
#plt.legend()
#plt.grid()
#plt.xlim(20,150)
#plt.ylim(-2.,1)

#plt.show()



