#!/usr/bin/env python
import argparse
import numpy as np
import scipy.optimize
import scipy.integrate
import configparser
from numba import jit
from compute_sigma8 import growth_factor
from compute_hod import differential_pair_count_NFW, differential_pair_count_NFW_samec

delta_c = 1.686
Delta_vir = 200.

Mpc_to_cm = 3.0856e24 # Conversion factor from Mpc to cm
Msun_to_g = 1.989e33 # Conversion factor from Msun to grams
G = 6.672e-8 # Universal Gravitational Constant in cgs units
Hubble = 3.2407789e-18 # Hubble's constant h/sec
rho_crit = (3.0*Hubble**2 / (8.0 * np.pi * G)) * (Mpc_to_cm**3 / Msun_to_g) # Msun h^2 / Mpc^3

def compute_power(x,y,z,weight, boxsize, ngrid=512):

	"""compute power spectrum from particles.x, particles.y, particles.z."""

	import pyximport; pyximport.install()
	import powerspec
	from compute_powerspec import power_from_particles_weights
	
	k=np.array([])
	pk=np.array([])
	nmodes=np.array([])

	## allocate arrays, plan FFT

	print('setting up FFT...',end='',flush=True)
	rhogrid, fft_of_rhogrid, fft_plan = powerspec.plan_fft_double(ngrid)
	print('done.')

	## compute power spectrum
	k, pk, nmodes = power_from_particles_weights(np.asarray(x,dtype=np.float64),
												 np.asarray(y,dtype=np.float64),
												 np.asarray(z,dtype=np.float64),
												 np.asarray(weight,dtype=np.float64),
												 ngrid,boxsize,rhogrid,fft_of_rhogrid,fft_plan,
												 subtract_shot_noise=True)

	return k, pk

@jit
def count_satsat_pairs(x,y,z,rbins):
	"""count pairs x,y,z into bins rbins."""
	r2bins = rbins**2
	points = np.zeros( (x.shape[0]) * (x.shape[0] - 1) )
	k = 0
	for i in range(x.shape[0]):
		for j in range(i):
			dx = x[i]-x[j]
			dy = y[i]-y[j]
			dz = z[i]-z[j]
			r2 = dx**2 + dy**2 + dz**2
			points[k] = r2
			k += 1
	counts, bins = np.histogram(points, bins=r2bins)

	return counts


def count_censat_pairs(x,y,z,rbins):
	r2bins = rbins**2
	points = x**2 + y**2 + z**2
	counts, bins = np.histogram(points,bins=r2bins)

	return counts
	

def monte_carlo_1halo(rbins, Ncen_of_M, Nsat_of_M,
						cM = None, dndm=None, Omega_M=None, redshift=None, mass_tab=None):
	"""compute monte carlo pair counts for 1-halo term."""

	ss_counts_array = np.zeros(rbins.shape[0])
	cs_counts_array = np.zeros(rbins.shape[0])
	nhalos = int(1e6)
	halo_mass = np.zeros(nhalos)
	#ncen = np.zeros(nhalos)
	nsat = np.zeros(nhalos)

	N_of_M = lambda M: Ncen_of_M(M) * (1.0 + Nsat_of_M(M))
	vec_NM = np.vectorize(N_of_M)
	ngal_integrand = dndm*vec_NM(mass_tab)
	ngal = scipy.integrate.simps(ngal_integrand*mass_tab, x=np.log(mass_tab))
	print("ngal = {}".format(ngal))


	"""step 1. sample in bins, then weight pair counts according to mass function."""

	mf = scipy.interpolate.interp1d(mass_tab, dndm)
	minmass = mass_tab.min()
	maxmass = mass_tab.max()
	halo_mass = np.logspace(np.log10(minmass), np.log10(maxmass), nhalos+1)
	halo_mass[0] = minmass
	halo_mass[-1] = maxmass
	halo_weight = mf(halo_mass[:-1]) * np.diff(halo_mass)


	"""--> step 2a. for each halo, populate galaxies according to Nsat_of_M."""

	vec_Ncen = np.vectorize(Ncen_of_M)
	ncen_exp = vec_Ncen(halo_mass[:-1])
	ncen = np.zeros(nhalos, dtype=np.uint64)
	ncen[np.random.uniform(size=nhalos) < ncen_exp] = 1

	vec_Nsat = np.vectorize(Nsat_of_M)
	nsat_exp = vec_Nsat(halo_mass[:-1])
	nsat = np.random.poisson(lam=nsat_exp)

	nsat_nonzero = nsat[np.nonzero(nsat)]
	ncen_nonzero = ncen[np.nonzero(nsat)]
	halo_mass_nonzero = halo_mass[np.nonzero(nsat)]
	halo_weight_nonzero = halo_weight[np.nonzero(nsat)]

	def NFW_cdf(x, cvir):
			"""the cumulative mass profile of an NFW halo.
				x = r/Rvir. cvir = Rvir/rs."""
			prefac = 1.0 / ( np.log( 1.0 + cvir ) - (cvir / ( 1.0 + cvir )) )
			return prefac * ( np.log( 1.0 + x * cvir ) - (x * cvir / ( 1.0 + x*cvir )) )

	def NFW_cdf_sample(xin, cvir):
			x = np.linspace(0., 1., 1000)
			invcdf = scipy.interpolate.interp1d(NFW_cdf(x,cvir), x)
			return invcdf(xin)

	num_gals = np.sum( ( ncen + (ncen*nsat) ) * halo_weight )
	print("num_gals = {}".format(num_gals))
	print("ncen = {}".format(np.sum( ncen * halo_weight ) / num_gals ))
	print("nsat = {}".format(np.sum( (ncen*nsat) * halo_weight ) / num_gals ))

	for k in range(nsat_nonzero.shape[0]):
		this_mass = halo_mass_nonzero[k]
		this_weight = halo_weight_nonzero[k]
		this_nsat = nsat_nonzero[k]
		this_ncen = ncen_nonzero[k]

		if this_ncen == 1:
			this_cvir = cM(this_mass)
			this_rvir = Rvir(this_mass, Omega_M)

			u = np.random.uniform(low=0.,high=1.,size=this_nsat)
			costheta = 2.0*np.random.uniform(size=this_nsat) - 1.0 # [-1,1) interval
			phi = 2.0*np.pi*np.random.uniform(size=this_nsat)

			r = Rvir(this_mass, Omega_M) * NFW_cdf_sample(u, this_cvir)
			sintheta = np.sqrt(1.0 - costheta*costheta)
			x = r*sintheta*np.cos(phi)
			y = r*sintheta*np.sin(phi)
			z = r*costheta
		
			"""--> step 2b. count satellite-satellite pairs within each halo,
				accumulate in count_array."""
			ss_counts_array[:-1] += count_satsat_pairs(x,y,z,rbins) * this_weight

			"""--> step 2c. count central-satellite pairs within each halo.
							(assume that whenever satellites exists, a central
							also exists.)"""
			cs_counts_array[:-1] += count_censat_pairs(x,y,z,rbins) * this_weight

	print(ss_counts_array)
	print(cs_counts_array)
	DD = ss_counts_array[:-1] + cs_counts_array[:-1]

	rmax = rbins[1:]
	rmin = rbins[0:-1]
	r  = 0.5*(rmax+rmin)

	RR = ngal * num_gals * (2./3.)*np.pi*(rmax**3 - rmin**3)
	xi = (DD/RR) - 1.0
	xi_cs = (cs_counts_array[:-1]/RR) - 1.0
	xi_ss = (ss_counts_array[:-1]/RR) - 1.0

	return r, xi, xi_cs, xi_ss
	
	
if __name__ == '__main__':
	# compute Zheng formula
	c = 5.0
	r = np.logspace(-3, 0., 50)
	zpairs = np.zeros(r.shape[0])
	spairs = np.zeros(r.shape[0])
	for i in range(r.shape[0]):
		zpairs[i] = differential_pair_count_NFW(r[i], c, c)
		spairs[i] = differential_pair_count_NFW_samec(r[i], c)
		print("{} {} {}".format(r[i],zpairs[i],spairs[i]))
	
	import matplotlib.pyplot as plt
	plt.figure()
	plt.plot(r, zpairs, label='zheng')
	plt.plot(r, spairs, label='sheth')
	plt.xlabel('radius/Rvir')
	plt.ylabel('pair counts')
	plt.legend(loc='best')
	plt.tight_layout()
	plt.savefig('hod_paircount_test.pdf')
	
	if monte_carlo:
		rmc, xi_1halo_mc, xi_cs_mc, xi_ss_mc = monte_carlo_1halo(rbins, Ncen_of_M, Nsat_of_M,\
				 cM=cM_interpolated, dndm=massfun_tab,  mass_tab = mass_tab,
				 Omega_M=omega_m, redshift=redshift)
