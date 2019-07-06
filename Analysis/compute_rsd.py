import numpy as np
import scipy.interpolate as interpolate
import scipy.integrate as integrate
from scipy.special import j0 as J0
from scipy.special import j1 as J1
import matplotlib.pyplot as plt
from numba import jit

import compute_wp
from compute_sigma8 import growth_factor
from compute_pk import compute_pk


@jit
def binavg_J0(kp, rp_min, rp_max):
	return 2.0*((rp_max*J1(kp*rp_max) - rp_min*J1(kp*rp_min)) / kp) / (rp_max**2 - rp_min**2)


@jit
def binavg_cos(kz, pi_min, pi_max):
	return ((np.sin(kz*pi_max) - np.sin(kz*pi_min)) / kz) / (pi_max - pi_min)


def compute_rsd_fftlog(k_in, P_in, D_growth, b_g, rpmin=None, rpmax=None,
					   pimin=None, pimax=None):

	"""compute linear-order rsd correction for wp with FFTLOG algorithm."""

	


@jit
def compute_rsd_correction(k_in, P_in, D_growth, b_g,
						   rpmin=None, rpmax=None, pimin=None, pimax=None):

	"""compute xi(rp, pi) with the linear Kaiser formalism with pk_gg as input."""
	
	
	## convert to uniform sampling in log-k for k<1, uniform in k for k>=1
	
	P_interp = interpolate.interp1d(k_in, P_in, kind='cubic')
	kin_max = k_in[-1]
	Pin_max = P_in[-1]
	
	
	## determine kmax
	
	eps = 1.0e-3
	kz_min = kp_min = 1e-4
	k_fid = 1.0
	kz_max = kp_max = k_fid * (eps / P_interp(k_fid))**(-1./3.)
	
	nlogbins = 768
	k_switch = 1.0
	assert kz_max > k_switch
	assert kp_max > k_switch
	kz_logbins = np.logspace(np.log10(kz_min), np.log10(k_switch), nlogbins+1)[:-1]
	kp_logbins = np.logspace(np.log10(kp_min), np.log10(k_switch), nlogbins+1)[:-1]
	kz_range = kz_max - k_switch
	kp_range = kp_max - k_switch
	
	
	## do Riemann sums
	
	beta = D_growth / b_g

	wpgg_rsd = np.zeros_like(rpmin)
	wpgg_norsd = np.zeros_like(rpmin)
	
	for i, (rp1, rp2) in enumerate(zip(rpmin, rpmax)):
		for j, (pi1, pi2) in enumerate(zip(pimin, pimax)):
			
			rp = 0.5*(rp1+rp2)
			pi = 0.5*(pi1+pi2)
			dpi = pi2-pi1
			
			kz_wavelength = (2.0*np.pi)/pi
			kp_wavelength = (2.0*np.pi)/rp
			kz_minbins = 768
			kp_minbins = 768
			oversample = 1.0
			kz_nbins = np.maximum(np.ceil(oversample*2.*(kz_range/kz_wavelength)), kz_minbins)
			kp_nbins = np.maximum(np.ceil(oversample*2.*(kp_range/kp_wavelength)), kp_minbins)
			kz_linbins = np.linspace(k_switch, kz_max, kz_nbins+1)
			kp_linbins = np.linspace(k_switch, kp_max, kp_nbins+1)
			kz_bins = np.concatenate( (kz_logbins, kz_linbins) )
			kp_bins = np.concatenate( (kp_logbins, kp_linbins) )
			
			dkp = np.diff(kp_bins)
			dkz = np.diff(kz_bins)
			kzs = 0.5*(kz_bins[1:] + kz_bins[:-1])
			kps = 0.5*(kp_bins[1:] + kp_bins[:-1])
			
			P_integrand = np.zeros_like(kps)
			
			for z in range(kzs.shape[0]):
			
				kz = kzs[z]
				kp = kps[:]
				ksq = kp**2 + kz**2
				musq = kz**2 / ksq
				absk = np.sqrt(ksq)
				P_integrand[absk< kin_max] = P_interp(absk[absk <kin_max])
				P_integrand[absk>=kin_max] = Pin_max*(absk[absk>=kin_max]/kin_max)**(-3.0)
	
				norsd_prefac = ( 1.0 / (2.0*np.pi**2) ) * kp * P_integrand
				rsd_prefac   = norsd_prefac * ( 1 + beta*musq )**2
				cos_fac = binavg_cos(kz,pi1,pi2) * binavg_J0(kp,rp1,rp2)
				
				rsd_integrand = rsd_prefac * cos_fac
				norsd_integrand = norsd_prefac * cos_fac
				
				wpgg_rsd[i] += 2.0 * dpi* np.sum(dkp * dkz[z] * rsd_integrand)
				wpgg_norsd[i] += 2.0 * dpi * np.sum(dkp * dkz[z] * norsd_integrand)
			
	return wpgg_rsd, wpgg_norsd

	
if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('input_xigg')
	parser.add_argument('input_ximm')
	parser.add_argument('output_correction')
	args = parser.parse_args()
	
	
	## compute power spectrum
	
	rgg_min, rgg_max, _, xigg = np.loadtxt(args.input_xigg, unpack=True)
	rmm_min, rmm_max, _, ximm = np.loadtxt(args.input_ximm, unpack=True)
	
	k, P_gg = compute_pk(rgg_min, rgg_max, xigg)
	k, P_mm = compute_pk(rmm_min, rmm_max, ximm)
	
	interp_gg = interpolate.interp1d(k, P_gg, kind='quadratic')
	interp_mm = interpolate.interp1d(k, P_mm, kind='quadratic')
		
	
	## estimate large-scale bias from sqrt( P_gg / P_mm )
	
	k_lin = np.pi / np.max(rgg_max)
	bias = np.sqrt( interp_gg(k_lin) / interp_mm(k_lin) )
		
	print(f"large-scale bias = {bias} at k = {k_lin}")
	
	
	## compute RSD correction
	
	Omega_m = 0.3
	z_lens = 0.3
	
	binmin = 0.1		# Mpc/h
	binmax = 32.5428	# Mpc/h
	N_rpbins = 19
	pimax = 100.0	# Mpc/h
	
	N_pibins = 20
	pibins = np.linspace(0., pimax, N_pibins+1)
	dpi = pibins[1] - pibins[0]
	
	rp_binmin, rp_binmax, wp_norsd_config = compute_wp.wp(rgg_min, rgg_max, xigg,
												   Omega_m, z_lens,
												   pimax=pimax,
					 							   rp_min=binmin, rp_max=binmax,
					 							   nbins=N_rpbins)
					 							   
	D_growth = growth_factor(redshift=z_lens, omega_m=Omega_m)
	   
	wp_rsd, wp_norsd = compute_rsd_correction(k, P_gg, D_growth, bias,
									rpmin = rp_binmin, rpmax = rp_binmax,
									pimin = pibins[:-1], pimax = pibins[1:])

	rsd_correction = wp_rsd / wp_norsd
	
	np.savetxt(args.output_correction, np.c_[rp_binmin, rp_binmax, np.zeros_like(rsd_correction), rsd_correction])
	
	
	print(f"wp_rsd = {wp_rsd}")
	print(f"wp_norsd = {wp_norsd}")
	print(f"wp_norsd (config. space) = {wp_norsd_config}")
	print(f"rsd_correction = {rsd_correction}")

	reference_rsd_correction = np.array([ 1.00572621, 1.00630931, 1.00721187, 1.00880302, 1.00964763, 1.01074008, 1.01207824, 1.01310299, 1.01514488, 1.01822484, 1.0212971,  1.02190469, 1.02422397, 1.03465635, 1.04566479, 1.05085479, 1.07876624, 1.10081198, 1.14770846])
	
	print(f"residuals = {rsd_correction - reference_rsd_correction}")
	assert(np.allclose(rsd_correction, reference_rsd_correction, atol=1e-3, rtol=1e-3))
	
	
#	plt.figure()
#	im = plt.imshow(np.swapaxes(np.log10(wp_rsd), 0, 1), origin='lower', cmap='viridis')
#	cbar = plt.colorbar(im)
#	plt.tight_layout()
#	plt.savefig('rsd_wp_plot.pdf')
	
	
#	plt.figure()
#	plt.plot(0.5*(rp_binmin+rp_binmax), wp_rsd, label='rsd')
#	plt.plot(0.5*(rp_binmin+rp_binmax), wp_norsd, label='no rsd')
#	plt.plot(0.5*(rp_binmin+rp_binmax), wp_norsd_config, label='no rsd (config. space)')
#	plt.yscale('log')
#	plt.xscale('log')
#	plt.xlabel('rp (Mpc/h)')
#	plt.ylabel('wp')
#	plt.legend(loc='best')
#	plt.tight_layout()
#	plt.savefig('rsd_wp_comparison_plot.pdf')
	