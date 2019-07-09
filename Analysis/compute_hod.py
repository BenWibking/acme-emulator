#!/usr/bin/env python

import argparse
import configparser
import config
import numpy as np
import scipy.optimize
import scipy.integrate
import scipy.signal
import camb
from compute_sigma8 import wcdm_growth_factor
from scipy.integrate import simps as simpson
from math import sqrt, log, log10, sin
from pathlib import Path	
from numba import jit

delta_c = 1.686
Delta_vir = 200.

Mpc_to_cm = 3.0856e24 # Conversion factor from Mpc to cm
Msun_to_g = 1.989e33 # Conversion factor from Msun to grams
G = 6.672e-8 # Universal Gravitational Constant in cgs units
Hubble = 3.2407789e-18 # Hubble's constant h/sec
rho_crit = (3.0*Hubble**2 / (8.0 * np.pi * G)) * (Mpc_to_cm**3 / Msun_to_g) # Msun h^2 / Mpc^3


def eisenstein_hu_pk(omch2=None, ombh2=None, sigma8=None, H0=None, w0=None, ns=None,
				redshift=None, do_nonlinear=False):
	
	"""compute Eisenstein & Hu (1998) fitting function for linear power spectrum."""

	from colossus.cosmology import power_spectrum

	h = H0/100.
	omega_m = (omch2+ombh2)/h**2
	omega_b = ombh2/h**2
	T_cmb = 2.7255	# Kelvins
	
	k = np.logspace(np.log10(1e-5), np.log10(1e3), 6000)
	T = power_spectrum.transferFunction(k, h, omega_m, omega_b, T_cmb,
										model='eisenstein98_zb')
	P = T**2 * k**ns

	## rescale P(k) according to D(z) and fix any normalization problems in input P(k)
	
	input_sigma_8 = sigma_R(k, P, R=8.0)
	h = H0/100.

	growth_factor_at_redshift = wcdm_growth_factor(redshift, omega_m=omega_m, w0=w0)
	rescale_factor = (growth_factor_at_redshift * sigma8 / input_sigma_8)**2
	P *= rescale_factor

	return k, P


#def camb_linear_pk(omch2=None, ombh2=None, sigma8=None, H0=None, w0=None, ns=None,
#				redshift=None, do_nonlinear=False):
#
#	camb_params = camb.CAMBparams()
#	camb_params.set_dark_energy(w=w0)
#	
#	camb_params.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0)
#	camb_params.InitPower.set_params(ns=ns, As=2.1e-9)
#	camb_params.set_matter_power(redshifts=[0.], kmax=1e3)
#
#	if do_nonlinear == True:
#		camb_params.NonLinear = camb.model.NonLinear_pk
#	else:
#		camb_params.NonLinear = camb.model.NonLinear_none
#
#	results = camb.get_results(camb_params)
#
#	k, z, pk = results.get_matter_power_spectrum(minkh=1e-5, maxkh=1e3, npoints=2048)
#	P_camb = pk[0,:]
#
#	# convert to uniform sampling in log-k (CAMB output is *not* log-spaced!)
#	log_k_camb = np.log10(k)
#	P_interp = scipy.interpolate.interp1d(log_k_camb, P_camb)
#	
#	# extrapolate past k_camb.max()
#	logkmax_extrapolate = 3.0
#	nsamples = 6000
#
#	logkmax = np.log10(k[-1])
#	Pmax = P_camb[-1]
#	P_asymp = lambda logk: Pmax * (10.**(-3.0*(logk-logkmax)))
#	
#	def P_interp_and_asymp(logk):
#		if logk >= logkmax:
#				return P_asymp(logk)
#		else:
#				return P_interp(logk)
#
#	P_vec = np.vectorize(P_interp_and_asymp)
#					
#	k = np.logspace(log_k_camb.min(), logkmax_extrapolate, nsamples)
#	log_k = np.log10(k)
#	dlogk = np.log(k[1]/k[0]) # natural log here!
#	P = P_vec(log_k)
#
#	## rescale P(k) according to D(z) and fix any normalization problems in input P(k)
#	
#	input_sigma_8 = sigma_R(k, P, R=8.0)
#	h = H0/100.
#	omega_m = (omch2+ombh2)/h**2
#
#	growth_factor_at_redshift = wcdm_growth_factor(redshift, omega_m=omega_m, w0=w0)
#	rescale_factor = (growth_factor_at_redshift * sigma8 / input_sigma_8)**2
#	P *= rescale_factor
#
#	return k, P


def compute_BAO_damping(k, P, Sigma=5.0):
	P_damped = P * np.exp( -(k*Sigma)**2 / 2.0 )
	return P_damped


def j1(x):
    return ( (np.sin(x)/x**2) - (np.cos(x)/x) )


def bin_avg_spherical_j0(k,rminus,rplus):

    """compute the bin-averaged spherical Bessel function j0."""
    
    integral = lambda r: r**2 * j1(k*r) / k
    return (3.0 / (rplus**3 - rminus**3)) * (integral(rplus) - integral(rminus))


def xi_binaverage(k_in, pk_in, binmin, binmax):
    pk_interp = scipy.interpolate.interp1d(k_in,pk_in)
    super_fac = 32
    k  = np.logspace(np.log10(k_in[0]),np.log10(k_in[-1]),k_in.shape[0]*super_fac)
    pk = pk_interp(k)

    bins = zip(binmin, binmax)
    xi = np.empty(binmin.shape[0])
    for i, (rminus, rplus) in enumerate(bins):
        # compute signal in bin i on the interval [rminus, rplus)
        y = k**2 / (2.0*np.pi**2) * bin_avg_spherical_j0(k,rminus,rplus) * pk
        result = scipy.integrate.simps(y*k, x=np.log(k)) # do integral in d(ln k)
        xi[i] = result
    return xi


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

	rho_mean = (Omega_M * rho_crit) # comoving units!
	return sigma, f_sigma, rho_mean/M, this_dln_sigma_inv_dM


def dndm_tinker(M,z=None,k=None,P=None,Omega_M=None):
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

	## N.B.: how you integrate the number density affects the s-s term by ~10% (!)
#	ngal = scipy.integrate.simps(integrand, x=M)
	ngal = np.sum(integrand[:-1]*dM)

	return ngal


def compute_HOD_parameters(ngal=None, M1_over_Mmin=None, M0_over_M1=None, alpha=None,
						siglogM=None, f_cen=None,
						mass_tabulated=None, massfun_tabulated=None, logMmin_guess=13.5):
						
	"""Compute the comoving units (Msun h^-1) HOD parameters from ratios of masses
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
	
	return logMmin, np.log10(M0), np.log10(M1)
				

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


def differential_pair_count_NFW_Zheng07(x, c1, c2):

	"""from Appendix A of Zheng & Weinberg (2007)."""

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

	return F


def differential_pair_count_NFW(x,c1,c2):
	#F = differential_pair_count_NFW_Zheng07(x,c1,c2)
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

	Fprime = lambda x, c: differential_pair_count_NFW(x, c, c)

	vec_NFW_profile = np.vectorize(NFW_profile)
	vec_Fprime = np.vectorize(Fprime)

	for i, r in enumerate(rbins):
		
		def counts_cs_integrand(logM):
		
			M = np.exp(logM)
			rvir = Rvir(M,Omega_M)
			mf = dndm

			# N.B.: I did something inconsistent in my weighed HOD scheme for the sims...
			#   [uncomment the first line below to make it consistent with my weighted HODs.]
			npairs_cs = ( vec_Ncen(M) * (vec_Ncen(M) * vec_Nsat(M)) ) * vec_NFW_profile(r/rvir, cM(M))
#			npairs_cs = ( vec_Ncen(M) * vec_Nsat(M) ) * vec_NFW_profile(r/rvir, cM(M))

			result = mf * (npairs_cs)/(rvir) * M
			return result

		def counts_ss_integrand(logM):
		
			M = np.exp(logM)
			rvir = Rvir(M,Omega_M)
			mf = dndm
			npairs_ss = ( 0.5 * vec_Ncen(M) * (vec_Nsat(M))**2 ) * vec_Fprime(r/rvir, cM(M))
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
					redshift=None, Omega_M=None, biasfun=None):

	"""compute 2-halo term from an HOD, bias-mass relation, and a linear Pk."""

	xi_mm = np.zeros(rbins.shape[0])
	bias = biasfun

	vec_NM = np.vectorize(N_of_M)
	bias_integrand = bias*dndm*vec_NM(mass_tab) 
	ngal_integrand = dndm*vec_NM(mass_tab)
	ngal = scipy.integrate.simps(ngal_integrand*mass_tab, x=np.log(mass_tab))
	b_g = scipy.integrate.simps(bias_integrand*mass_tab, x=np.log(mass_tab)) / ngal

	## compute xi_mm from P(k)
	
	dr_over_r = 0.005
	binmin = rbins * (1.0 - dr_over_r)
	binmax = rbins * (1.0 + dr_over_r)
	xi_mm = xi_binaverage(k, Pk, binmin, binmax)

	return rbins, b_g**2 * xi_mm, xi_mm
	
	
def compute_xigm_1halo(rbins, N_of_M, Ncen_of_M, Nsat_of_M, 
						cM = None, dndm=None, Omega_M=None, redshift=None, mass_tab=None):

	"""compute 1halo power spectrum given a function N(M) 'N_of_M' that gives the HOD:
	P_gg,1h(k) = (1/n_g^2) \int_0^\inf 0.5*(N_of_M(M)*(N_of_M(M)-1)) W^2(k,M) (dn/dM) dM
	"""

	xi_gm = np.zeros(rbins.shape[0])
	xi_gm_cm = np.zeros(rbins.shape[0])
	xi_gm_sm = np.zeros(rbins.shape[0])
	vec_NM = np.vectorize(N_of_M)
	vec_Nsat = np.vectorize(Nsat_of_M)
	vec_Ncen = np.vectorize(Ncen_of_M)
	ngal_integrand = dndm*vec_NM(mass_tab)
	ngal = scipy.integrate.simps(ngal_integrand, x=mass_tab)

	Fprime = lambda x, c: differential_pair_count_NFW(x, c, c)

	vec_NFW_profile = np.vectorize(NFW_profile)
	vec_Fprime = np.vectorize(Fprime)

	rho_mean = (Omega_M * rho_crit) # comoving units!

	for i, r in enumerate(rbins):
		
		def counts_cm_integrand(logM):
			M = np.exp(logM)
			rvir = Rvir(M,Omega_M)
			mf = dndm
			npairs_cm = ( vec_Ncen(M) * M ) * vec_NFW_profile(r/rvir, cM(M))
			result = mf * (npairs_cm)/(rvir) * M
			return result

		def counts_sm_integrand(logM):
			M = np.exp(logM)
			rvir = Rvir(M,Omega_M)
			mf = dndm
			npairs_sm = ( vec_Ncen(M) * vec_Nsat(M) * M ) * vec_Fprime(r/rvir, cM(M))
			result = mf * (npairs_sm)/(rvir) * M
			return result

		DD_cm = scipy.integrate.simps(counts_cm_integrand(np.log(mass_tab)),x=np.log(mass_tab))
		DD_sm = scipy.integrate.simps(counts_sm_integrand(np.log(mass_tab)),x=np.log(mass_tab))
		RR = 2.0*np.pi*r**2 * ngal * rho_mean
		xi_gm_cm[i] = DD_cm/RR
		xi_gm_sm[i] = DD_sm/RR

		xi_gm[i] = ((DD_cm + DD_sm)/RR)

	return rbins, xi_gm, xi_gm_cm, xi_gm_sm


def compute_xigm_2halo(rbins, N_of_M, Ncen_of_M, Nsat_of_M, k=None, Pk=None, dndm=None, mass_tab=None, 
						redshift=None, Omega_M=None, biasfun=None):

	"""compute 2-halo term from an HOD, bias-mass relation, and a linear Pk."""

	xi_mm = np.zeros(rbins.shape[0])
	bias = biasfun

	vec_NM = np.vectorize(N_of_M)
	bias_integrand = bias*dndm*vec_NM(mass_tab) 
	ngal_integrand = dndm*vec_NM(mass_tab)
	ngal = scipy.integrate.simps(ngal_integrand*mass_tab, x=np.log(mass_tab))
	b_g = scipy.integrate.simps(bias_integrand*mass_tab, x=np.log(mass_tab)) / ngal

	## compute xi_mm from P(k)
	dr_over_r = 0.005
	binmin = rbins * (1.0 - dr_over_r)
	binmax = rbins * (1.0 + dr_over_r)
	xi_mm = xi_binaverage(k, Pk, binmin, binmax)

	## assume r_gm == 1
	return rbins, b_g * xi_mm, xi_mm
	
	
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


def elementwise_integral_secondorder(rp, binmin, binmax, xi, pimax):

	lower_bound = rp
	upper_bound = np.sqrt(rp**2 + pimax**2)

	# offset bins by 0.5*dr
	bin_median = 0.5*(binmin+binmax)
	bin_minus = bin_median[:-1]
	bin_plus = bin_median[1:]
	binmask = np.logical_and(bin_plus > lower_bound, bin_minus < upper_bound)
	xi_minus = xi[:-1][binmask]
	xi_plus = xi[1:][binmask]
	r_minus = bin_minus[binmask]
	r_plus = bin_plus[binmask]
	# integration limits may lie within a bin, need to be careful
	s_minus = np.maximum(lower_bound, r_minus)
	s_plus = np.minimum(upper_bound, r_plus)

	# here we assume that xi is piecewise linear over the tabulated input bins
	m = (xi_plus - xi_minus) / (r_plus - r_minus)
	const_term = 2.0*(xi_minus - m*r_minus) * \
				(np.sqrt(s_plus**2 - rp**2) - np.sqrt(s_minus**2 - rp**2))
	linear_term = m * ( s_plus*np.sqrt(s_plus**2 - rp**2) - \
						s_minus*np.sqrt(s_minus**2 - rp**2) + \
						rp**2 * np.log( (s_plus + np.sqrt(s_plus**2 - rp**2)) / \
										(s_minus + np.sqrt(s_minus**2 - rp**2)) ) )
	integral = linear_term + const_term
	
	return np.sum(integral)


def wp(binmin, binmax, xi, pimax=100.0, rp_binmin=None, rp_binmax=None):

	"""compute w_p(r_p) from tabulated xi(r)."""

	w_p = np.zeros(rp_binmin.shape[0])

	for i,(this_rp_binmin,this_rp_binmax) in enumerate(zip(rp_binmin,rp_binmax)):
		rp = 0.5*(this_rp_binmin + this_rp_binmax)
		w_p[i] += elementwise_integral_secondorder(rp,binmin,binmax,xi,pimax)

	return rp_binmin, rp_binmax, w_p
	
	
if __name__=='__main__':
                              
	parser = argparse.ArgumentParser()
	
	parser.add_argument('header_file')	# defines cosmological parameters
	parser.add_argument('pk_file')
	parser.add_argument('massfun_file')
	parser.add_argument('halobias_file')

	parser.add_argument('hod_params_path')
	
	parser.add_argument('xigg_input_file')
	parser.add_argument('xigg_output_file')
	parser.add_argument('xigg_ratio_output_file')
	
	parser.add_argument('wp_input_file')
	parser.add_argument('wp_output_file')
	parser.add_argument('wp_ratio_output_file')
	
	args = parser.parse_args()


	## read HOD parameters
	
	myconfigparser = configparser.ConfigParser()
	myconfigparser.read(args.hod_params_path)
	params = myconfigparser['params']

	f_cen = float(params['f_cen'])
	siglogM = float(params['siglogM'])
	alpha = float(params['alpha'])
	input_ngal = float(params['ngal'])
	M0_over_M1 = float(params['m0_over_m1'])
	M1_over_Mmin = float(params['m1_over_mmin'])
	
#	print("HOD parameters:",file=sys.stderr)
#	print("\tinput_ngal = {}".format(input_ngal),file=sys.stderr)
#	print("\tM1_over_Mmin = {}".format(M1_over_Mmin),file=sys.stderr)
#	print("")


	## read cosmological params, compute linear Pk with CAMB
	
	header_file = args.header_file

	cf = config.AbacusConfigFile(header_file)
	omega_m = cf.Omega_M	# at z=0
	redshift = cf.redshift
	sigma_8 = cf.sigma_8
	ns = cf.ns
	ombh2 = cf.ombh2
	omch2 = cf.omch2
	w0 = cf.w0		## WARNING: growth factor is currently computed assuming w == -1.0!!!
	H0 = cf.H0


#	## compute (linear) power spectrum
#	
#	k, P = camb_linear_pk(ombh2=ombh2, omch2=omch2, H0=H0, ns=ns, w0=w0,
#							sigma8=sigma_8, redshift=redshift)
#
#
#	## convenience functions for mass function fitting formulae
#
#	dndm = lambda M: dndm_tinker(M, z=redshift, k=k, P=P, Omega_M=omega_m)
#	bias = lambda M: compute_linear_bias(M, k, P, omega_m=omega_m)
#	
#	dndm_vec = np.vectorize(dndm)
#	bias_vec = np.vectorize(bias)
#
#	mass_tab = np.logspace(10., 16., 512)
#	massfun_tab = dndm_vec(mass_tab)
#
#	bias_mass_tab = mass_tab
#	bias_tab = bias_vec(bias_mass_tab)


	## load cosmological-dependent quantities from files
	
	k, P = np.loadtxt(args.pk_file, unpack=True)
	mass_tab, massfun_tab = np.loadtxt(args.massfun_file, unpack=True)
	bias_mass_tab, bias_tab = np.loadtxt(args.halobias_file, unpack=True)


	## find HOD parameters

	logMmin, logM0, logM1 = compute_HOD_parameters(ngal=input_ngal,
													siglogM=siglogM,
													M0_over_M1=M0_over_M1,
													M1_over_Mmin=M1_over_Mmin,
													alpha=alpha,
													f_cen=f_cen,
													mass_tabulated=mass_tab,
													massfun_tabulated=massfun_tab)
													

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

	
	## compute correlation functions

	in_binmin, in_binmax, err_xigg_sim, xigg_sim = np.loadtxt(args.xigg_input_file,
															  unpack=True)

	rbins = 0.5*(in_binmin + in_binmax)

	cM_interpolated = lambda M: cM_Correa2015(M, z=redshift)

	r, xi_gg1, xi_gg1cs, xi_gg1ss = compute_xi_1halo(rbins, global_N_of_M,
					 Ncen_of_M, Nsat_of_M,
					 cM=cM_interpolated, dndm=massfun_tab, mass_tab=mass_tab,
					 Omega_M=omega_m, redshift=redshift)

	r2, xi_gg2, xi_mm = compute_xi_2halo(rbins, global_N_of_M,
						redshift=redshift, k=k, Pk=P,
						dndm=massfun_tab, mass_tab=bias_mass_tab,
						Omega_M=omega_m, biasfun=bias_tab)


#	## truncate 2-halo term at ~1 Mpc/h
#	r_cut = 2.0 # Mpc/h
#	rcut_decay = 0.3 # Mpc/h
#	xi_gg2[r2 <= r_cut] *= np.exp( -0.5 * ( (r_cut - r2[r2 <= r_cut]) / rcut_decay )**2 )

	xi = xi_gg1 + xi_gg2


	## output xi_gg correlation function
	
	np.savetxt(args.xigg_output_file, np.c_[in_binmin, in_binmax, np.zeros(xi.shape), xi])
	

	## output xi_gg_sim(r) / xi_gg_analytic(r)

	xigg_ratio = xigg_sim / xi
#	xigg_ratio = scipy.signal.savgol_filter(xigg_ratio, 9, 2)  # 9-point quadratic fit
	xigg_ratio = scipy.signal.savgol_filter(xigg_ratio, 3, 0)  # 3-point moving average
	err_xigg_ratio = err_xigg_sim / xi # fix this

	np.savetxt(args.xigg_ratio_output_file, np.c_[in_binmin, in_binmax,
												  err_xigg_ratio, xigg_ratio])


	## output wp(r_p)

	in_rp_binmin, in_rp_binmax, err_wp_sim, wp_sim = np.loadtxt(args.wp_input_file, unpack=True)
	
	rp_binmin, rp_binmax, this_wp = wp(rbins, rbins, xi, pimax=100.0,
										rp_binmin=in_rp_binmin, rp_binmax=in_rp_binmax)
										
	np.savetxt(args.wp_output_file, np.c_[rp_binmin, rp_binmax, np.zeros(this_wp.shape[0]), this_wp])
	
	
	## output wp_sim(r_p) / wp_analytic(r_p)
	
	wp_ratio = wp_sim / this_wp
	err_wp_ratio = err_wp_sim / this_wp
	
	np.savetxt(args.wp_ratio_output_file, np.c_[in_rp_binmin, in_rp_binmax, err_wp_ratio, wp_ratio])
	
