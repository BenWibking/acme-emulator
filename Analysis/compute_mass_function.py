#!/usr/bin/env python

import argparse
import numpy as np
import h5py as h5

import config


def mass_function(mass_list, boxsize, nbins=4096):

	"""
	Compute the mass function dn/dM for the input array of halo masses in bins of log mass
	"""
	
#    mmin_halos = mass_list.min()
#    mmax_halos = mass_list.max()
	
	mmin_halos = 10.**(10)		# Msun/h
	mmax_halos = 10.**(15.5)	# Msun/h
		
	assert( mmin_halos > 0. )	# one of the catalogs has a halos with m_SO == 0.0

	M_bins = np.logspace(np.log10(mmin_halos), np.log10(mmax_halos), nbins+1)
	bin_counts, bin_edges = np.histogram(mass_list, bins=M_bins)

	vol = boxsize**3
	dM = np.diff(M_bins)
	mf = bin_counts / vol / dM
	binmin = bin_edges[:-1]
	binmax = bin_edges[1:]
	
	return binmin, binmax, mf
	
	
def compute_mass_function(halo_file, header_file, output_file):

	cf = config.AbacusConfigFile(header_file)
	boxsize = cf.boxSize
	assert( boxsize > 0. )

	with h5.File(halo_file, mode='r') as catalog:
		halos = catalog['halos']
		binmin, binmax, mass_fun = mass_function(halos['mass'], boxsize)

	## save mass function
	
	assert( ~np.any(np.isnan(binmin)) )
	assert( ~np.any(np.isnan(binmax)) )
	assert( ~np.any(np.isnan(mass_fun)) )
	
	np.savetxt(output_file, np.c_[binmin, binmax, mass_fun])


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('header_path')
	parser.add_argument('halo_path')
	parser.add_argument('output_path')

	args = parser.parse_args()

	compute_mass_function(args.halo_path, args.header_path, args.output_path)

