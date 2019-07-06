#!/usr/bin/env python
import argparse
import numpy as np
import scipy.optimize
import scipy.integrate
import sys
import configparser
from numba import jit
from compute_sigma8 import growth_factor
from scipy.integrate import simps as simpson


delta_c = 1.686
Delta_vir = 200.

Mpc_to_cm = 3.0856e24 # Conversion factor from Mpc to cm
Msun_to_g = 1.989e33 # Conversion factor from Msun to grams
G = 6.672e-8 # Universal Gravitational Constant in cgs units
Hubble = 3.2407789e-18 # Hubble's constant h/sec
rho_crit = (3.0*Hubble**2 / (8.0 * np.pi * G)) * (Mpc_to_cm**3 / Msun_to_g) # Msun h^2 / Mpc^3


def camb_linear_pk(omch2=None, ombh2=None, sigma8=None, H0=None, w0=None, ns=None,
				redshift=None):
	import camb

	camb_params = camb.CAMBparams()
	camb_params.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0)
	camb_params.set_dark_energy(w=w0)
	camb_params.InitPower.set_params(ns=ns, As=2.1e-9)
	camb_params.set_matter_power(redshifts=[0.], kmax=1e3)
	camb_params.NonLinear = camb.model.NonLinear_none
	#camb_params.NonLinear = camb.model.NonLinear_pk
	results = camb.get_results(camb_params)

	k, z, pk = results.get_matter_power_spectrum(minkh=1e-5, maxkh=200., npoints=2048)
	P = pk[0,:]

	# rescale P(k) according to D(z) and fix any normalization problems in input P(k)
	input_sigma_8 = sigma_R(k, P, R=8.0)
	h = H0/100.
	omega_m = (omch2+ombh2)/h**2
	growth_factor_at_redshift = growth_factor(redshift=redshift, omega_m=omega_m)
	rescale_factor = (growth_factor_at_redshift * sigma8 / input_sigma_8)**2
	P *= rescale_factor

	return k, P


def W(k,r):
	return 3.0*( np.sin(k*r) - (k*r)*np.cos(k*r) ) / (k*r)**3
	

def sigma_R(k, P, R=None):
	"""compute rms variance of overdensity in spheres of radius R (Mpc h^-1)."""
	integrand = P * k**2 * W(k,R)**2 / (2 * np.pi**2)
	this_sigma_R = np.sqrt(simpson(integrand, x=k))
	return this_sigma_R


def sigma_M(k, P, M, Omega_M):
	mean_rho = (rho_crit*Omega_M) # in comoving units
	R = ((3.0 * M) / (4.0*np.pi*mean_rho))**(1./3.)	# no delta_vir
	return sigma_R(k, P, R=R)


def Rvir(M, Omega_M):
	mean_rho = (rho_crit*Omega_M) # in comoving units
	return ((3.0 * M) / (4.0*np.pi*Delta_vir*mean_rho))**(1./3.)


def dln_sigma_inv_dM(k, P, M=None, Omega_M=None):
	"""compute d ln sigma^-1 / dM analytically."""

	mean_rho = (rho_crit*Omega_M) # in comoving units

	def dW_dM(k,R):
		x = k * R
		# no Delta_vir
		return k * M**(-2./3.) * (3.0/(4.0*np.pi*mean_rho))**(1./3.) * \
					(np.sin(x)/x**2 + 3.0*np.cos(x)/x**3 - 3.0*np.sin(x)/x**4)

	R = ((3.0 * M) / (4.0*np.pi*mean_rho))**(1./3.) # no Delta_vir
	integrand = P * k**2 * (2.0 * W(k,R) * dW_dM(k,R)) / (2.0 * np.pi**2)

	this_dln_sigma_R_inv_dM = simpson(integrand, x=k)

	return this_dln_sigma_R_inv_dM
	

def compute_peak_height(m_halo, k, P, omega_m):
	"""compute peak height for a halo of mass m_halo given a (z=0) power spectrum P(k).
	sigma^2 = (1/2pi^2) \int P(k) W^2(kR) k^2 dk."""

	this_sigma_M = sigma_M(k, P, m_halo, omega_m)
	nu = delta_c / this_sigma_M

	return nu


def compute_linear_bias(m_halo, k, P, omega_m=None):
	"""compute the linear bias for halos of mass logMmin
		based on the Tinker+ 2010 fitting function."""

	nu = compute_peak_height(m_halo, k, P, omega_m)

	## fitting function (eq. 6 from Tinker et al. 2010)

	y = np.log10(Delta_vir)
	A = 1.0 + ( 0.24 * y * np.exp(-(4.0/y)**4) )
	a = (0.44 * y) - 0.88
	B = 0.183
	b = 1.5
	C = 0.019 + (0.107 * y) + ( 0.19 * np.exp(-(4.0/y)**4) )
	c = 2.4

	bias = 1.0 - ( A * (nu**a / (nu**a + delta_c**a)) ) + (B * nu**b) + (C * nu**c)
	return bias


def compute_linear_bias_ying(m_halo, k, P, omega_m=None):
	""" ying's code. """

	y = np.log10(Delta_vir)
	y44 = np.exp(-np.power(4.0/y, 4.0))
	bias_A = 1.00 + 0.24*y*y44
	bias_a = (y-2.0)*0.44
	bias_B = 0.4
	bias_b = 1.5
	bias_C = ((y-2.6)*0.4 + 1.11 + 0.7*y*y44)*0.94
	bias_c = 2.4

	sig = sigma_M(k, P, m_halo, omega_m)
	a = np.power(sig, -bias_a)
	b = (1.0 - bias_A * a / (a + 1.0) + bias_B * np.power(sig, -bias_b) 
									  + bias_C * np.power(sig, -bias_c))

	return b


def dndm_tinker_all(M, z=None, k=None, P=None, Omega_M=None):
	"""analytic fitting formula for mass function from Tinker+ 2008
		as a function of redshift and Delta_vir."""

	sigma = sigma_M(k, P, M=M, Omega_M=Omega_M)

	A = 0.186 * ((1.0+z)**(-0.14))
	a = 1.47 * ((1.0+z)**(-0.06))
	alpha = 10.**( - ( (0.75/np.log10(Delta_vir/75.))**(1.2) ) )
	b = 2.57 * ((1.0+z)**(-alpha))
	c = 1.19
	f_sigma = A * ( (sigma / b)**(-a) + 1.0 ) * np.exp(-c/(sigma**2))

	## compute ( d ln sigma^{-1} / dM )
	this_dln_sigma_inv_dM = dln_sigma_inv_dM(k, P, M=M, Omega_M=Omega_M) / (-2.0*sigma**2)

	## compute finite difference approximation
	#eps = 1.01	# for computing finite differences of (ln sigma^(-1))
	#sigma_plus = sigma_M(k, P, M=M*eps, Omega_M=Omega_M)
	#this_dln_sigma_inv = np.log(1.0/sigma_plus) - np.log(1.0/sigma)
	#this_dM = M*eps - M
	#finite_this_dln_sigma_inv_dM = this_dln_sigma_inv / this_dM
	#frac_err = finite_this_dln_sigma_inv_dM/this_dln_sigma_inv_dM - 1.0
	#if abs(frac_err) > 0.005:
	#	print("INCORRECT DERIVATIVE: {}".format(frac_err))

	rho_mean = (Omega_M * rho_crit) # comoving units!
	return sigma, f_sigma, rho_mean/M, this_dln_sigma_inv_dM


def dndm_tinker(M,z=None,P=None,Omega_M=None):
	sigma, f_sigma, rho_mean_over_M, this_dln_sigma_inv_dM = dndm_tinker_all(M,z,k,P,Omega_M)
	return f_sigma*(rho_mean_over_M)*this_dln_sigma_inv_dM


def compute_ngal(logM, dM, mass_function,
				 logMmin, M1_over_Mmin, M0_over_M1, alpha, siglogM, f_cen):
	"""compute number density from HOD."""

	M = 10.**logM
	Mmin = 10.**logMmin
	M1 = M1_over_Mmin*Mmin
	M0 = M0_over_M1*M1

	mean_cen = 0.5 * (1. + scipy.special.erf((logM - logMmin)/siglogM))
	mean_sat = np.zeros(M.shape[0])
	mean_sat[M > M0] = mean_cen[M > M0] * (((M[M > M0] - M0) / M1)**alpha)

	integrand = ( (mean_cen * f_cen) + mean_sat ) * mass_function

	## how you integrate the number density affects the s-s term by ~10% (!)
#	ngal = scipy.integrate.simps(integrand, x=M)
	ngal = np.sum(integrand[:-1]*dM)

	return ngal


def compute_HOD_parameters(ngal=None, M1_over_Mmin=None, M0_over_M1=None, alpha=None,
						siglogM=None, f_cen=None,
						mass_tabulated=None, massfun_tabulated=None, logMmin_guess=13.5):
	"""Compute the physical (Msun h^-1) HOD parameters from ratios of masses
	and the desired number density of galaxies."""

	dM = np.diff(mass_tabulated)
	M = mass_tabulated
	mass_function = massfun_tabulated
	logM = np.log10(M)

	this_ngal = lambda logMmin: compute_ngal(logM, dM, mass_function, logMmin,
											M1_over_Mmin, M0_over_M1, alpha, siglogM, f_cen)
	objective = lambda logMmin: this_ngal(logMmin) - ngal
	logMmin = scipy.optimize.newton(objective, logMmin_guess, maxiter = 100)

	Mmin = 10**(logMmin)
	M1 = Mmin*M1_over_Mmin
	M0 = M1*M0_over_M1
	
	print('logMmin: %s' % logMmin, file=sys.stderr)
	print('optimized ngal: %s' % this_ngal(logMmin), file=sys.stderr)
	print('desired ngal: %s\n' % ngal, file=sys.stderr)

	return logMmin, np.log10(M0), np.log10(M1)


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
					this_rvir = Rvir(this_mass, Omega_M, redshift)

					u = np.random.uniform(low=0.,high=1.,size=this_nsat)
					costheta = 2.0*np.random.uniform(size=this_nsat) - 1.0 # [-1,1) interval
					phi = 2.0*np.pi*np.random.uniform(size=this_nsat)

					r = Rvir(this_mass, Omega_M, redshift) * NFW_cdf_sample(u, this_cvir)
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
				

def differential_pair_count_NFW_samec(s, c):
	"""compute differential pair count at separation s = r/Rvir
		using Sheth et al. 2001 [MNRAS 325, 1288-1302] formula.
		[c = Rvir/rs]."""

	a = 1.0/c
	x = s*c		# == r/a (in units of Rvir)
	prefac = 1.0 / ( np.log( 1.0 + c ) - (c / ( 1.0 + c )) )
	norm = a**3 / prefac**2
	res = 0.

	if s <= 1.0:
		t1 = ( -4.0*(1.0+a) + 2.0*a*x*(1.0+2.0*a) + a**2 * x**2 ) / \
				( 2.0*x**2 * (1.0+a)**2 * (2.0+x) )
		t2 = x**(-3.0) * np.log( ( (1.0+a-a*x)*(1.0+x) ) / (1.0+a) )
		t3 = np.log(1.0+x) / ( x*(2.0+x)**2 )
		res = t1+t2+t3

	elif s > 1.0 and s <= 2.0:
		t1 = np.log( (1.0+a)/(a*x+a-1.0) ) / ( x*(2.0+x)**2 )
		t2 = ( a**2 * x - 2.0*a ) / ( 2.0*x*(1.0+a)**2 * (2.0+x) )
		res = t1+t2

	return (1.0/norm) * res * s**2


def differential_pair_count_NFW_allterms(x, c1, c2):
	"""from Appendix A of Zheng & Weinberg (2007)."""

	from math import sqrt, log, log10, sin
	A_0 = 2.4575
	alpha = -3.099
	beta = 0.617
	c_T = 1.651
	mu = 4.706
	B_0 = 0.0336
	omega = 2.684
	phi = 0.4079
	Astar = lambda c: A_0 * c**(3.0+alpha) * (1.0 + (c/c_T)**((beta-alpha)/mu))**mu * \
					 (1.0 + B_0*sin(omega*(log10(c)-phi)))
	A = lambda c1,c2: sqrt(Astar(c1)*Astar(c2))

	s = 2.0*x

	f1=0.
	f2=0.
	f3=0.
	f4=0.
	f=0.

	if x <= 0.5:
		f1 = 1.0/(c2+c1+c1*c2*s)**2 * log((1.0+c1*s)*(1.0+c2*s)) + \
			 c1*s/(c2*(c2+c1+c1*c2*s)*(1.0+c1*s))

		eps = 1.0e-5
		if abs(s - (c1-c2)/(c1*c2)) < eps:
			f2 = (c1**(-2) - c2**(-2) * (1.0+c1)**(-2))/2.0
		else:
			f2 = 1.0/(c2-c1+c1*c2*s)**2 * log((1.0+c1*s)*(1.0+c2-c2*s)/(1.0+c1)) - \
					c1*(1.0-s)/(c2*(c2-c1+c1*c2*s)*(1.0+c1*s)*(1.0+c1))

		if abs(s - (c2-c1)/(c1*c2)) < eps:
			f3 = (c1**(-2) * (1.0+c2)**(-2) - c2**(-2))/2.0
		else:
			f3 = 1.0/(c2-c1-c1*c2*s)**2 * log((1.0+c2*s)*(1.0+c1-c1*s)/(1.0+c2)) + \
				 c1*(1.0-s)/(c2*(c2-c1-c1*c2*s)*(1.0+c1-c1*s))

		f4 = -s/(c2*(1.0+c1)*(1.0+c2)*(1.0+c1-c1*s))

		F = A(c1,c2)*( f1+f2+f3+f4 )*s

	elif x <= 1.0:
		f = 1.0/(c2+c1+c1*c2*s)**2 * log((1.0+c1)*(1.0+c2)/((1.0-c1+c1*s)*(1.0-c2+c2*s))) + \
			 (s-2.0)/((1.0+c1)*(1.0+c2)*(c2+c1+c1*c2*s))
		F = A(c1,c2)*f*s

	elif x > 1.0:
		F = 0.0


	if F < 0.0:
		print("negative Fprime({},{},{}) = {}".format(x,c1,c2,F))

	return F,f1,f2,f3,f4,f


def differential_pair_count_NFW(x,c1,c2):
	#F,f1,f2,f3,f4,f = differential_pair_count_NFW_allterms(x,c1,c2)
	F = differential_pair_count_NFW_samec(x,c1)
	return F


def cM_Correa2015(M,z=None):
	"""concentration-mass relation from Correa et al. (2015)"""

	logM = np.log10(M)
	alpha = 1.62774 - 0.2458*(1.0 + z) + 0.01716*(1.0 + z)**(2.0)
	beta = 1.66079 + 0.00359*(1.0 + z) - 1.6901*(1.0 + z)**(0.00417)
	gamma = -0.02049 + 0.0253*(1.0 + z)**(-0.1044)
	exponent = alpha + beta*logM*(1.0 + gamma*logM**(2.0))

	# Approximate factor (\sqrt 2) to rescale Rvir between crit, matter
	cvir = np.sqrt(2.0) * 10.0**(exponent) 

	return cvir


#def cM(M,z=None):
#	return cM_Correa2015(M,z=z)



def NFW_profile(R, cvir):
	"""R = r/rvir, cvir = rvir / rs, x = r/rs."""
	x = R * cvir
	if R < 1.0:
		prefac = 1.0 / ( np.log( 1.0 + cvir ) - (cvir / ( 1.0 + cvir )) )
		return prefac * ( cvir * x**2 / ( x * ( 1.0 + x )**2 ) )
	else:
		return 0.


def compute_xi_1halo(rbins, N_of_M, Ncen_of_M, Nsat_of_M, 
						cM = None, dndm=None, Omega_M=None, redshift=None, mass_tab=None):
	"""compute 1halo power spectrum given a function N(M) 'N_of_M' that gives the HOD:
	P_gg,1h(k) = (1/n_g^2) \int_0^\inf 0.5*(N_of_M(M)*(N_of_M(M)-1)) W^2(k,M) (dn/dM) dM
	"""

	xi_gg = np.zeros(rbins.shape[0])
	xi_gg_cs = np.zeros(rbins.shape[0])
	xi_gg_ss = np.zeros(rbins.shape[0])
	vec_NM = np.vectorize(N_of_M)
	vec_Nsat = np.vectorize(Nsat_of_M)
	vec_Ncen = np.vectorize(Ncen_of_M)
	ngal_integrand = dndm*vec_NM(mass_tab)
	ngal = scipy.integrate.simps(ngal_integrand, x=mass_tab)
	print("ngal = {}".format(ngal))

	Fprime = lambda x, c: differential_pair_count_NFW(x, c, c)

	vec_NFW_profile = np.vectorize(NFW_profile)
	vec_Fprime = np.vectorize(Fprime)

	for i, r in enumerate(rbins):
		
		def counts_cs_integrand(logM):
			M = np.exp(logM)
			rvir = Rvir(M,Omega_M)
			mf = dndm
			npairs_cs = ( vec_Ncen(M) * vec_Nsat(M) ) * vec_NFW_profile(r/rvir, cM(M))
				# if there is a satellite, there is always a central
			result = mf * (npairs_cs)/(rvir) * M
			return result

		def counts_ss_integrand(logM):
			M = np.exp(logM)
			rvir = Rvir(M,Omega_M)
			mf = dndm
			npairs_ss = ( 0.5 * (vec_Nsat(M))**2 ) * vec_Fprime(r/rvir, cM(M))
			result = mf * (npairs_ss)/(rvir) * M
			return result

		DD_cs = scipy.integrate.simps(counts_cs_integrand(np.log(mass_tab)),x=np.log(mass_tab))
		DD_ss = scipy.integrate.simps(counts_ss_integrand(np.log(mass_tab)),x=np.log(mass_tab))
		RR = 2.0*np.pi*r**2 * ngal**2
		xi_gg_cs[i] = DD_cs/RR
		xi_gg_ss[i] = DD_ss/RR
		xi_gg[i] = ((DD_cs + DD_ss)/RR)

	return rbins, xi_gg, xi_gg_cs, xi_gg_ss


def compute_xi_2halo(rbins, N_of_M, k=None, Pk=None, dndm=None, mass_tab=None, 
					redshift=None,
					Omega_M=None, biasfun=None):
	"""compute 2-halo term from an HOD, bias-mass relation, and a linear Pk."""

	xi_mm = np.zeros(rbins.shape[0])

	#bias = lambda M: compute_linear_bias(M, k, Pk, omega_m=Omega_M, sigma_8=sigma_8, redshift=redshift)
	bias = biasfun

	vec_NM = np.vectorize(N_of_M)
	bias_integrand = bias*dndm*vec_NM(mass_tab) 
	ngal_integrand = dndm*vec_NM(mass_tab)
	ngal = scipy.integrate.simps(ngal_integrand*mass_tab, x=np.log(mass_tab))
	b_g = scipy.integrate.simps(bias_integrand*mass_tab, x=np.log(mass_tab)) / ngal

	print("b_g = {}".format(b_g))


	## compute xi_mm from P(k)

	for i in range(rbins.shape[0]):
		r = rbins[i]
		hankel_pk = k**2 * Pk * (np.sin(k*r) / (k*r))
		xi_mm[i] = 1.0/(2.0*np.pi**2) * scipy.integrate.simps(hankel_pk, x=k)

	return rbins, b_g**2 * xi_mm, xi_mm
	

def main(args):
	monte_carlo = args.montecarlo


	## read HOD parameters

	myconfigparser = configparser.ConfigParser()
	myconfigparser.read(args.hod_params_path)
	params = myconfigparser['params']

	print("mass function: {}".format(args.mass_fun_path))

	f_cen = float(params['f_cen'])
	siglogM = float(params['siglogM'])
	alpha = float(params['alpha'])
	input_ngal = float(params['ngal'])
	M0_over_M1 = float(params['m0_over_m1'])
	M1_over_Mmin = float(params['m1_over_mmin'])
	print("HOD parameters:",file=sys.stderr)
	print("\tinput_ngal = {}".format(input_ngal),file=sys.stderr)
	print("\tM1_over_Mmin = {}".format(M1_over_Mmin),file=sys.stderr)


	## read cosmological params, compute linear Pk with CAMB

	#import config
	#cf = config.AbacusConfigFile(args.header_file)
	#omega_m = cf.OmegaNow_m # at z=redshift
	#redshift = cf.redshift
	#sigma_8 = cf.sigma_8
	#ns = cf.ns
	#ombh2 = cf.ombh2
	#omch2 = cf.omch2
	#w0 = cf.w0
	#H0 = cf.H0

	omegam_bolshoi = 0.27
	redshift_bolshoi = 0.1
	sigma8_bolshoi = 0.82
	H0_bolshoi = 70.
	ns_bolshoi = 0.95
	omegab_bolshoi = 0.0469
	h_bolshoi = H0_bolshoi/100.
	omegac_bolshoi = omegam_bolshoi-omegab_bolshoi
	w0_bolshoi = -1.0

	## (unneeded)
	#omegamnow_bolshoi = omegam_bolshoi*(1.0+redshift_bolshoi)**3 / \
	#					( (1.0 - omegam_bolshoi) + omegam_bolshoi*(1.0+redshift_bolshoi)**3 )
	#print("omega_mnow (bolshoi) = {}".format(omegamnow_bolshoi))

	omega_m = omegam_bolshoi
	redshift = redshift_bolshoi
	sigma_8 = sigma8_bolshoi
	ns = ns_bolshoi
	ombh2 = omegab_bolshoi*h_bolshoi**2
	omch2 = omegac_bolshoi*h_bolshoi**2
	w0 = w0_bolshoi
	H0 = H0_bolshoi
	h = H0/100.
	
	from colossus.cosmology import cosmology
	cosmo = cosmology.setCosmology('bolshoi')


	## read mass functions
	mass_tab, massfun_tab, bias_tab = np.loadtxt(args.mass_fun_path, unpack=True)
	
	#mass_tab *= h
	#massfun_tab *= h**(-4.)


	## convenience functions for mass function fitting formulae

	dndm = lambda M: dndm_tinker(M, z=redshift, k=k, P=P, Omega_M=omega_m)
	dndm_all = lambda M: dndm_tinker_all(M, z=redshift, k=k, P=P, Omega_M=omega_m)
	

	## find HOD parameters

	logMmin, logM0, logM1 = compute_HOD_parameters(ngal=input_ngal,
													siglogM=siglogM,
													M0_over_M1=M0_over_M1,
													M1_over_Mmin=M1_over_Mmin,
													alpha=alpha,
													f_cen=f_cen,
													mass_tabulated=mass_tab,
													massfun_tabulated=massfun_tab)
	print("logMmin: {}\nlogM0: {}\nlogM1: {}".format(logMmin,logM0,logM1),file=sys.stderr)


	## convert h units
	## TODO [*should* be done before finding HOD parameters, but this is to check a bug...]

	mass_tab *= h
	massfun_tab *= h**(-4.)

	import colossus.lss.mass_function as colossus_mf
	colossus_dndm = colossus_mf.massFunction(mass_tab, redshift,
											 mdef='200m', model='tinker08', q_out='dndlnM')
	colossus_dndm /= mass_tab


	## compute (linear) power spectrum
	
	#k, P = camb_linear_pk(ombh2=ombh2, omch2=omch2, H0=H0, ns=ns, w0=w0,
	#					sigma8=sigma_8, redshift=redshift)

	k = np.logspace(-5.0, 3.0, 4096)
	P = cosmo.matterPowerSpectrum(k,z=redshift,model='eisenstein98')
	print("redshift = {}".format(redshift))

	## plot NFW convolution profile

	import matplotlib.pyplot as plt
	plt.figure()

	x = np.linspace(1.0e-3, 2.0, 100)
	c = 10.0
	Fconv = np.zeros(x.shape)
	NFW = np.zeros(x.shape)
	for i in range(x.shape[0]):
		Fconv[i] = differential_pair_count_NFW(x[i], c, c)
		NFW[i] = NFW_profile(x[i], c)
	print("F_ss normalization = {}".format(scipy.integrate.simps(Fconv, x=x)))
	print("F_cs normalization = {}".format(scipy.integrate.simps(NFW, x=x)))
	print("F_ss+cs normalization = {}".format(scipy.integrate.simps(NFW + Fconv, x=x)))
	plt.plot(x, Fconv, label='NFW convolution')
	plt.plot(x, NFW, label='NFW')
	plt.legend(loc='best')
	plt.savefig('NFW_convolution.pdf')


	## plot halo bias functions

	ying_M, ying_massfun, ying_halobias = np.loadtxt('halostat_bolshoi.txt',unpack=True)
	tinker_halobias = np.zeros(ying_M.shape[0])
	ying_thiscode_halobias = np.zeros(ying_M.shape[0])

	for i in range(ying_M.shape[0]):
		tinker_halobias[i] = compute_linear_bias(ying_M[i], k, P, omega_m=omega_m)
		ying_thiscode_halobias[i] = compute_linear_bias_ying(ying_M[i], k, P, omega_m=omega_m)

	import colossus.lss.bias as colossus_bias
	colossus_halobias = colossus_bias.haloBias(ying_M, redshift, model='tinker10', mdef='200m')

	import matplotlib.pyplot as plt
	plt.figure()

	plt.plot(ying_M, ying_halobias, label='ying')
	plt.plot(ying_M, tinker_halobias, label='tinker')
	#plt.plot(ying_M, colossus_halobias, label='colossus')
	plt.plot(ying_M, ying_thiscode_halobias, '--', label='ying (my code)')
	#plt.plot(ying_M, tinker_halobias/ying_halobias, label='ratio to ying')
	#plt.plot(ying_M, tinker_halobias/colossus_halobias, label='ratio to colossus')
	#plt.plot(ying_M, tinker_halobias/ying_thiscode_halobias, '--', label='ratio to ying (my code)')

	plt.xlim(10.**(12.75), 10.**(15.))
	plt.ylim(1., 9.)
	plt.xscale('log')
	plt.xlabel(r'halo mass ($M_{\odot} \, h^{-1}$)')
	plt.legend(loc='best')
	plt.title('bias function comparison')
	plt.savefig('analytic_halobias.pdf')


	## plot halo mass function

	tinker_massfun = np.zeros(mass_tab.shape[0])
	tinker_sigma = np.zeros(mass_tab.shape[0])
	tinker_f_sigma = np.zeros(mass_tab.shape[0])
	tinker_dlnsigmainv_dM = np.zeros(mass_tab.shape[0])
	peak_height = np.zeros(mass_tab.shape[0])

	for i in range(mass_tab.shape[0]):
		sigma, f_sigma, rho_mean_over_M, dlnsigmainv_dM = dndm_all(mass_tab[i]) 
		peak_height[i] = compute_peak_height(mass_tab[i], k, P, omega_m)
		tinker_sigma[i] = sigma
		tinker_massfun[i] = f_sigma*rho_mean_over_M*dlnsigmainv_dM
		tinker_f_sigma[i] = f_sigma
		tinker_dlnsigmainv_dM[i] = dlnsigmainv_dM

	# w1_mass, w1_mf = np.loadtxt('hmf_wibking_h1.0.txt', unpack=True)
	# w07_mass, w07_mf = np.loadtxt('hmf_wibking_h0.7.txt', unpack=True)

	import matplotlib.pyplot as plt
	plt.figure()

	#plt.plot(mass_tab, massfun_tab, label='input dn/dM')
	#plt.plot(mass_tab, tinker_massfun, label='tinker formula dn/dM')
	#plt.plot(mass_tab, colossus_dndm, label='colossus')

	import colossus.lss.peaks as peaks
	colossus_peak_height = peaks.peakHeight(mass_tab, redshift)

	#plt.plot(mass_tab, peak_height, label=r'$\nu(M)$')
	#plt.plot(mass_tab, colossus_peak_height, label=r'$\nu(M)$ (colossus)')

	#plt.plot(mass_tab, tinker_f_sigma, label=r'tinker $f(\sigma)$')
	#plt.plot(mass_tab, tinker_dlnsigmainv_dM, label=r'tinker $ d \ln f^{-1} / dM$')

	plt.plot(mass_tab, tinker_massfun/massfun_tab, '--', alpha=0.5, label='ratio to ying') 
	plt.plot(mass_tab, tinker_massfun/colossus_dndm, '--', alpha=0.5, label='ratio to colossus') 

	plt.xlim(1.0e10, 5.0e15)
	plt.xscale('log')
	#plt.yscale('log')
	plt.ylabel(r'$d n / d M$')
	plt.xlabel(r'halo mass ($M_{\odot} \, h^{-1}$)')
	plt.legend(loc='best')
	plt.title('mass function comparison')
	plt.savefig('analytic_Massfun.pdf')


	## compute 1-halo xi_gg

	M0 = 10.**(logM0)
	M1 = 10.**(logM1)

	def Ncen_of_M(M):
		return 0.5 * (1.0 + scipy.special.erf((np.log10(M) - logMmin) / siglogM))
		
	def Nsat_of_M(M):
		Nsat = 0.
		if M > M0:
				#Nsat = Ncen_of_M(M) * ( ((M - M0)/M1)**alpha )
				Nsat = ( ((M - M0)/M1)**alpha )
		return Nsat

	def global_N_of_M(M):
		Ncen = Ncen_of_M(M)
		Nsat = Nsat_of_M(M)
		return Ncen * (1.0 + Nsat)


	## plot N(M)

	plt.figure()
	M = np.logspace(10., np.log10(5.0e15), 50)
	ncen_tab = np.zeros(M.shape)
	nsat_tab = np.zeros(M.shape)
	for i in range(M.shape[0]):
		ncen_tab[i] = Ncen_of_M(M[i])
		nsat_tab[i] = Nsat_of_M(M[i])
	plt.plot(M, ncen_tab, label='ncen')
	plt.plot(M, nsat_tab, label='nsat (no factor of Ncen included)')
	hodtab_mass, hodtab_ncen, hodtab_nsat = np.loadtxt('hod_wibking.txt', unpack=True)
	plt.plot(hodtab_mass, hodtab_ncen, label='ying ncen')
	plt.plot(hodtab_mass, hodtab_nsat, label='ying nsat')
	plt.ylim(0.01, 100.)
	plt.xlim(1e12, 5e15)
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('halo mass ($M_{\odot} \, h^{-1}$)')
	plt.ylabel('halo occupation')
	plt.legend(loc='best')
	plt.savefig('analytic_HOD.pdf')

	
	## compute correlation functions

	rbins = np.logspace(-2, 2, 30) # r bins

	cm_mass, cm_cvir = np.loadtxt(args.cm_relation_path, unpack=True)
	cM_interpolated = scipy.interpolate.interp1d(cm_mass, cm_cvir, bounds_error=False,
													fill_value=(cm_cvir[0],cm_cvir[-1]))

	r, xi_gg1, xi_gg1cs, xi_gg1ss = compute_xi_1halo(rbins, global_N_of_M,
					 Ncen_of_M, Nsat_of_M,
					 cM=cM_interpolated, dndm=massfun_tab, mass_tab=mass_tab,
					 Omega_M=omega_m, redshift=redshift)
	
	xigg1cs_integral = scipy.integrate.trapz(xi_gg1cs*r**2, x=r)
	print("xigg1cs_integral = {}".format(xigg1cs_integral))

	if monte_carlo:
		rmc, xi_1halo_mc, xi_cs_mc, xi_ss_mc = monte_carlo_1halo(rbins, Ncen_of_M, Nsat_of_M,\
				 cM=cM_interpolated, dndm=massfun_tab,  mass_tab = mass_tab,
				 Omega_M=omega_m, redshift=redshift)

	r2, xi_gg2, xi_mm = compute_xi_2halo(rbins, global_N_of_M, redshift=redshift, k=k, Pk=P,\
						dndm=massfun_tab, mass_tab=mass_tab, Omega_M=omega_m, biasfun=bias_tab)


	## plot correlation functions

	import matplotlib.pyplot as plt
	plt.figure()

	#	plt.plot(r,xi_gg1,'--',label='1-halo')
	plt.plot(r,xi_gg1cs,'-.',label='1-halo central-satellite')
	plt.plot(r,xi_gg1ss,'--',label='1-halo satellite-satellite')

	if monte_carlo:
		plt.plot(rmc,xi_1halo_mc,'-.',label='1-halo monte carlo (cs+ss)')
		plt.plot(rmc,xi_cs_mc,'-.',label='1-halo monte carlo (cs)')
		plt.plot(rmc,xi_ss_mc,'-.',label='1-halo monte carlo (ss)')
		print(xi_1halo_mc)

	plt.plot(r2,xi_gg2,label='2-halo analytic')
	

	def interpolate_or_nan(x,y):
		interpolator = scipy.interpolate.interp1d(x,y)
		xmin = np.min(x)
		xmax = np.max(x)
		def interp_fun(z):
				if z >= xmin and z <= xmax:
						return interpolator(z)
				else:
						return np.NaN
		return np.vectorize(interp_fun)	
		
	xi = xi_gg1 + xi_gg2
	# plt.plot(r, xi, '-', label='analytic HOD', color='black')

	if args.comparison_plot:
		r_comp,xi_comp,xi_cs_comp,xi_ss_comp,xi_2h_comp = np.loadtxt(args.comparison_plot, unpack=True)
		#plt.plot(r_comp, xi_comp, label='comparison function', color='red')
		plt.plot(r_comp, xi_cs_comp, '--', label='1-halo cs comparison', color='red')
		xi_cs_comp_integral = scipy.integrate.trapz(xi_cs_comp*r_comp**2, x=r_comp)
		print("xi_cs_comp_integral = {}".format(xi_cs_comp_integral))
		plt.plot(r_comp, xi_ss_comp, '-.', label='1-halo ss comparison', color='red')
		plt.plot(r_comp, xi_2h_comp-1.0, '--', label='2-halo comparison', color='red')

	plt.legend(loc='best')
	plt.xscale('log')
	plt.yscale('log')
	plt.ylim(0.1, 1.0e5)
	plt.xlim(rbins.min(), rbins.max())
	plt.savefig('analytic_xi_gg.pdf')

	if args.comparison_plot:
		xi_comp_interp = interpolate_or_nan(r_comp, xi_comp)
		frac_diff = (xi/xi_comp_interp(r)) - 1.0

		xi_cs_comp_interp = interpolate_or_nan(r_comp, xi_cs_comp)
		xi_ss_comp_interp = interpolate_or_nan(r_comp, xi_ss_comp)
		frac_diff_cs = (xi_gg1cs/xi_cs_comp_interp(r)) - 1.0
		frac_diff_ss = (xi_gg1ss/xi_ss_comp_interp(r)) - 1.0

		plt.figure()
		plt.plot(r, frac_diff, label='comparison function', color='red')

		plt.plot(r, frac_diff_cs, label='central-satellite', color='green')
		plt.plot(r, frac_diff_ss, label='satellite-satellite', color='blue')

		plt.xscale('log')
		plt.ylim(-0.4, 0.4)
		plt.legend(loc='best')
		plt.savefig('analytic_frac_diff.pdf')
		

if __name__=='__main__':                               
	parser = argparse.ArgumentParser()

	# parser.add_argument('header_file')
	parser.add_argument('cm_relation_path')
	parser.add_argument('mass_fun_path')
	parser.add_argument('bias_fun_path')
	parser.add_argument('hod_params_path')
	parser.add_argument('--comparison_plot')
	parser.add_argument('--centrals_only',default=False,action='store_true')
	parser.add_argument('--montecarlo',default=False,action='store_true')

	args = parser.parse_args()
	main(args)

