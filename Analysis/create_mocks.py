#!/usr/bin/env python

import argparse
import numpy as np
import scipy
import h5py as h5
import pandas as pd
import matplotlib.pyplot as plt
import os.path
from collections import defaultdict
import Corrfunc
from Corrfunc.theory.xi import xi as corrfunc_xi
from Corrfunc.theory.DD import DD as corrfunc_DD


def elementwise_integral_secondorder(rp,binmin,binmax,xi,pimax):

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


def DeltaSigma(binmin, binmax, xi, pimax=None,
				Omega_m=None, Omega_m_fid=0.3, z_lens=None,
				rp_binmin=None, rp_binmax=None):

	"""compute DeltaSigma (assuming the actual cosmology of the simulation)."""

	## compute mean_rho (comoving density units = Msun pc^-3)
	H0 = 100.
	speed_of_light_km_s = 2.998e5 # km/s
	csq_over_G = 2.494e12 # 3c^2/(8*pi*G) Msun pc^-1
	mean_rho = Omega_m * csq_over_G * (H0/speed_of_light_km_s)**2 / 1.0e12 # Msun pc^-3

	## compute rp bins
	rp_mid = (rp_binmin + rp_binmax)/2.0

	def _wp(rp,binmin,binmax,xi,pimax):
		   """compute wp(r_p) from tabulated xi(r)."""
		   return elementwise_integral_secondorder(rp,binmin,binmax,xi,pimax)

	ds = np.zeros(rp_mid.shape[0])
	integrand = lambda r: r*_wp(r,binmin,binmax,xi,pimax)
	for i in range(rp_mid.shape[0]):
		integral, abserr = scipy.integrate.quad(integrand, 0., rp_mid[i], epsabs=1.0e-3, epsrel=1.0e-3)
		ds[i] = (integral * (2.0/rp_mid[i]**2) - _wp(rp_mid[i],binmin,binmax,xi,pimax)) * mean_rho

	# convert Mpc/h unit to pc/h
	ds *= 1.0e6

	return rp_binmin, rp_binmax, ds


def main(args):

	"""Select subsamples from iHOD mock catalog."""

	## read galaxy catalog file

	catalog_file = h5.File('./data/iHODcatalog_mdr1.h5')
	catalog = catalog_file['galaxy']

	## Catalog fields:
	##	['conc', 'g-r', 'halo_id', 'lg_halo_mass', 'lg_stellar_mass', 'x', 'y', 'z', 'z_rs']

	color = catalog['g-r'][:]
	## If lg_halo_mass > 0., then it's a central galaxy. otherwise, it's a satellite
	lg_halo_mass = catalog['lg_halo_mass'][:]
	lg_stellar_mass = catalog['lg_stellar_mass'][:]
	x = catalog['x'][:]
	y = catalog['y'][:]
	z = catalog['z'][:]
	galaxy_halo_id = catalog['halo_id'][:]

	## read halo catalog file

	halo_file = pd.read_hdf('./data/mdr1_rockstar_host_200b_v3.h5')

	## Halo fields:
	##	['row_id', 'scale', 'rockstarId', 'x', 'y', 'z', 'M200b']

	all_halo_masses = halo_file['M200b'][:]
	lg_all_halo_masses = np.log10(all_halo_masses)
	all_halo_id = halo_file['rockstarId'][:]
	
	## create dictionary to look up M200b from rockstarId

	lookup_mass = defaultdict(lambda: 1.0, zip(all_halo_id, all_halo_masses))

	## Perform cuts on the galaxy catalog
	Mstar_fid = 11.2
	color_fid = 0.95
	stellarmass_scatter_dex = 0.25

	noisy_lg_stellar_mass = lg_stellar_mass + \
							stellarmass_scatter_dex * np.random.normal(size=lg_stellar_mass.shape[0])

	stellarmass_bins = np.linspace(10., 12.0, 50)
	color_bins = np.linspace(0.4, 1.2, 30)

	color_params = [-0.2, 0., 0.1]
	Mstar_params = [11.1, 11.1, 11.2]

	rps = []
	wps = []
	rmins = []
	rmaxs = []
	ravgs = []
	xis = []
	halomass_bin_list = []
	central_hod_list = []
	satellite_hod_list = []
	ndens_list = []
	sample_smfs = []

	for color_alpha, Mstar in zip(color_params, Mstar_params):

		## select sample

		mask = noisy_lg_stellar_mass > ( Mstar * (color / color_fid)**color_alpha )
		sample_lg_stellar_mass = noisy_lg_stellar_mass[mask]  # *observed* stellar masses
		sample_true_stellar_mass = lg_stellar_mass[mask] # *true* stellar masses
		sample_lg_halo_mass = lg_halo_mass[mask]
		sample_color = color[mask]
		sample_x = x[mask]
		sample_y = y[mask]
		sample_z = z[mask]
		sample_halo_id = galaxy_halo_id[mask]
		
		## compute statistics of the sample

		centrals_lg_halo_mass = sample_lg_halo_mass[sample_lg_halo_mass > 0.]
		sats_lg_halo_mass = np.log10(np.array(
					[lookup_mass[x] for x in sample_halo_id[np.logical_not(sample_lg_halo_mass > 0.)]]))

		ncen = centrals_lg_halo_mass.shape[0]
		nsat = sats_lg_halo_mass.shape[0]
		print("nsat: {}".format(nsat))
		ngal = sample_lg_stellar_mass.shape[0]
		fsat = 1.0 - (ncen / ngal)
		boxsize = 1000.
		vol = boxsize**3	# (Mpc/h)**3
		ndens = ngal / vol
		ndens_list.append(ndens)
		print("[Mstar: {}, alpha: {}] sample size = {} galaxies".format(Mstar,color_alpha, ngal))
		print("\t number density = {} (Mpc/h)^-3".format(ndens))
		print("\t satellite fraction = {}".format(fsat))

		## compute stellar mass functions (sample vs. total)

		lg_stellarmass_bins = np.linspace(10.6, 11.6, 20)
		sample_stellarmass_counts, mstar_bin_edges = np.histogram(sample_true_stellar_mass,
																  bins=lg_stellarmass_bins)
		total_stellarmass_counts, mstar_bin_edges = np.histogram(lg_stellar_mass,
																 bins=lg_stellarmass_bins)
		dM = np.diff(mstar_bin_edges)
		center_mstar_bins = 0.5*(mstar_bin_edges[1:] + mstar_bin_edges[:-1])
		sample_smf = center_mstar_bins * sample_stellarmass_counts / vol / dM
		total_smf = center_mstar_bins * total_stellarmass_counts / vol / dM
		sample_smfs.append(sample_smf)

		## compute HOD of centrals of this sample

		lg_halomass_bins = np.linspace(10., 15., 30)
		centrals_counts, halomass_bin_edges = np.histogram(centrals_lg_halo_mass, bins=lg_halomass_bins)
		halo_counts, halomass_bin_edges = np.histogram(lg_all_halo_masses, bins=lg_halomass_bins)
		with np.warnings.catch_warnings():
			np.warnings.filterwarnings('ignore')	# ignore division-by-zero errors
			centrals_hod = centrals_counts / halo_counts

		halomass_bin_centers = 0.5*(halomass_bin_edges[1:]+halomass_bin_edges[:-1])
		halomass_bin_list.append(halomass_bin_centers)
		central_hod_list.append(centrals_hod)

		## compute HOD of satellite in this sample

		satellite_counts, halomass_bin_edges = np.histogram(sats_lg_halo_mass, bins=lg_halomass_bins)
		satellite_hod = satellite_counts / halo_counts
		satellite_hod_list.append(satellite_hod)

		## compute clustering of this sample

		nthreads = 8
		binfile = os.path.abspath('./data/xigg_binfile.txt')
		xi_results = corrfunc_xi(boxsize, nthreads, binfile, sample_x, sample_y, sample_z)
		rmin = xi_results['rmin']
		rmax = xi_results['rmax']
		ravg = xi_results['ravg']
		xi = xi_results['xi']
		npairs = xi_results['npairs']
		r = 0.5*(rmin+rmax)
		rmins.append(rmin)
		rmaxs.append(rmax)
		ravgs.append(ravg)
		xis.append(xi)

		## compute projected clustering of sample and write to disk

		rpmin = 0.1
		rpmax = 30.0
		rp_bins = np.logspace(np.log10(rpmin), np.log10(rpmax), 30)
		rp_binmin = rp_bins[:-1]
		rp_binmax = rp_bins[1:]
		rp_binmin, rp_binmax, this_wp = wp(rmin, rmax, xi, pimax=100.,
										   rp_binmin=rp_binmin, rp_binmax=rp_binmax)
		rp = 0.5*(rp_binmin + rp_binmax)
		rps.append(rp)
		wps.append(this_wp)

		## compute \xi_gm for this sample

		# read in matter particle subsample
		
#		nthreads = 8
#		binfile = os.path.abspath('./data/xigg_binfile.txt') # same as xi_gg
#		xigm_results = corrfunc_DD(0, nthreads, binfile,
#									sample_x, sample_y, sample_z,
#									matter_x, matter_y, matter_z)
#		nbins = len(xigm_results)
#		rmingm = np.zeros(nbins)
#		rmaxgm = np.zeros(nbins)
#		DDgm = np.zeros(nbins)
#		RRgm = np.zeros(nbins)
#		for i in range(nbins):
#			rmingm[i] = xigm_results[i][0]
#			rmaxgm[i] = xigm_results[i][1]
#			DDgm[i] = xigm_results[i][3]  		# check this!!
#
#			RRgm[i] = ngals*nparticles*(4./3.)*np.pi*(rmax[i]**3 - rmin[i]**3) / vol
#
#		xigm = (DDgm / RRgm) - 1.0
#
#		## compute w_{p,gm} for this sample
#
#		rp_binmin, rp_binmax, this_wpgm = wp(rmingm, rmaxgm, xigm, pimax=100., rp_min=0.1, rp_max=30.0)
#		rp = 0.5*(rp_binmin + rp_binmax)
#		rp_gms.append(rp)
#		wp_gms.append(this_wpgm)

		print("")

	## plot stellar mass functions for each sample
	
	plt.figure()
	for color_alpha, sample_smf in zip(color_params, sample_smfs):
		plt.plot(center_mstar_bins, sample_smf,
				 label=r'$\alpha = {}$'.format(color_alpha))
	plt.plot(center_mstar_bins, total_smf, label=r'full catalog')
	plt.xlabel('stellar mass')
	plt.ylabel(r'$dn / d \ln M$')
	plt.yscale('log')
	plt.xlim(10.6, 11.6)
#	plt.ylim(1e-3, 1e-1)
	plt.legend(loc='best')
	plt.title('true stellar mass functions')
	plt.tight_layout()
	plt.savefig("./output/stellarmass_functions.pdf")
 
	## plot stellar mass-color histogram with selection cuts indicated

	stellarmass_color_hist, xbins, ybins = np.histogram2d(noisy_lg_stellar_mass, color,
												  bins=[stellarmass_bins, color_bins],
												  normed=True)
	plt.figure()
	X, Y = np.meshgrid(xbins, ybins)
	im = plt.pcolormesh(X, Y, stellarmass_color_hist.swapaxes(0,1),
						cmap=plt.get_cmap('magma'), linewidth=0., rasterized=True)

	for color_alpha in color_params:
		cut_stellarmass = Mstar * (color_bins / color_fid)**color_alpha
		plt.plot(cut_stellarmass, color_bins, '--', label=r'$\alpha={}$'.format(color_alpha))

	plt.colorbar(im)
	plt.xlim(stellarmass_bins.min(), stellarmass_bins.max())
	plt.ylim(color_bins.min(), color_bins.max())
	plt.xlabel(r'stellar mass')
	plt.ylabel(r'g-r color')
	plt.title(r'stellar mass $+$ color-selected galaxy samples')
	legend = plt.legend(loc='lower right')
	for t in legend.get_texts():
		t.set_color('white')
	plt.tight_layout()
	plt.savefig("./output/stellarmass_color_histogram.pdf")
	plt.close()
	
	## plot stellar mass-halo mass histogram
	
	halomass_bins = np.linspace(11., 14., 40)
	stellarmass_halomass_hist, xbins, ybins = np.histogram2d(lg_halo_mass, noisy_lg_stellar_mass,
															 bins=[halomass_bins, stellarmass_bins],
															 normed=True)
	plt.figure()
	X, Y = np.meshgrid(xbins, ybins)
	im = plt.pcolormesh(X, Y, stellarmass_halomass_hist.swapaxes(0,1),
						cmap=plt.get_cmap('magma'), linewidth=0., rasterized=True)
	plt.colorbar(im)
	plt.plot([10., 16.], [Mstar_fid, Mstar_fid], '--', color='white', label='fiducial stellar mass cut')
	plt.ylim(stellarmass_bins.min(), stellarmass_bins.max())
	plt.xlim(halomass_bins.min(), halomass_bins.max())
	plt.ylabel(r'stellar mass')
	plt.xlabel(r'halo mass')
	legend = plt.legend(loc='upper right')
	for t in legend.get_texts():
		t.set_color('white')
	plt.title(r'stellar mass vs. halo mass')
	plt.tight_layout()
	plt.savefig("./output/stellarmass_halomass_histogram.pdf")
	plt.close()

	## plot color-halo mass histogram
	
	color_halomass_hist, xbins, ybins = np.histogram2d(lg_halo_mass, color,
													   bins=[halomass_bins, color_bins],
													   normed=True)
	plt.figure()
	X, Y = np.meshgrid(xbins, ybins)
	im = plt.pcolormesh(X, Y, color_halomass_hist.swapaxes(0,1),
						cmap=plt.get_cmap('magma'), linewidth=0., rasterized=True)
	plt.colorbar(im)
	plt.plot([10., 16.], [color_fid, color_fid], '--', color='white', label='color pivot')
	plt.ylim(color_bins.min(), color_bins.max())
	plt.xlim(halomass_bins.min(), halomass_bins.max())
	plt.ylabel(r'color')
	plt.xlabel(r'halo mass')
	legend = plt.legend(loc='lower right')
	for t in legend.get_texts():
		t.set_color('white')
	plt.title(r'color vs. halo mass')
	plt.tight_layout()
	plt.savefig("./output/color_halomass_histogram.pdf")
	plt.close()

	## plot central HODs

	plt.figure()
	for massbins, central_hod, color_alpha in zip(halomass_bin_list,
													central_hod_list, color_params):
		plt.plot(massbins, central_hod, label=r'$\alpha = {}$'.format(color_alpha))
	plt.legend(loc='best')
	plt.xlabel(r'$\log_{10}$ halo mass')
	plt.ylabel('halo occupation')
	plt.title('central HOD')
	plt.tight_layout()
	plt.savefig('./output/central_hod_mocks.pdf')
	plt.close()

	## write central HODs to disk
	
	for massbins, central_hod, satellite_hod, color_alpha in zip(halomass_bin_list,
													central_hod_list,
													satellite_hod_list,
													color_params):
		linear_massbins = 10.**massbins
		central_hod[np.isnan(central_hod)] = 0.
		central_hod[np.isinf(central_hod)] = 0.
		satellite_hod[np.isnan(satellite_hod)] = 0.
		satellite_hod[np.isinf(satellite_hod)] = 0.
		output_hod_file = "./output/hod_alpha_{}.txt".format(color_alpha)
		np.savetxt(output_hod_file, np.c_[linear_massbins, central_hod, satellite_hod])

	## plot projected clustering

	plt.figure()
	for rp, this_wp, color_alpha in zip(rps, wps, color_params):
		plt.plot(rp, this_wp, label=r'$\alpha = {}$'.format(color_alpha))
	plt.legend(loc='best')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$r_p$ (Mpc/h)')
	plt.ylabel(r'$w_p$ (Mpc/h)')
	plt.title('projected correlation function')
	plt.tight_layout()
	plt.savefig('./output/wp_mocks.pdf')
	plt.close()

	## write xi to disk

	for rmin, rmax, ravg, xi, alpha in zip(rmins, rmaxs, ravgs, xis, color_params):
		output_file = './output/xi_alpha_{}.txt'.format(alpha)
		np.savetxt(output_file, np.c_[rmin, rmax, ravg, xi])

	## write wp to disk

	for rp, this_wp, color_alpha in zip(rps, wps, color_params):
		output_file = './output/wp_alpha_{}.txt'.format(color_alpha)
		np.savetxt(output_file, np.c_[rp, rp, np.zeros(rp.shape[0]), this_wp])

	## write ngal to disk
	
	for ndens, color_alpha in zip(ndens_list, color_params):
		output_file = './output/ngal_alpha_{}.txt'.format(color_alpha)
		np.savetxt(output_file, np.c_[ndens])


if __name__=='__main__':                               
	parser = argparse.ArgumentParser()

#	parser.add_argument('input_catalog')
#	parser.add_argument('output_file')

	args = parser.parse_args()
	main(args)

