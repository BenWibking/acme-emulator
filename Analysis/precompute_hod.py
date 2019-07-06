#!/usr/bin/env python

import numpy as np
import argparse
import compute_hod
import config


if __name__=='__main__':
                              
	parser = argparse.ArgumentParser()
	
	parser.add_argument('header_file')	# defines cosmological parameters

	parser.add_argument('output_pk_file')
	parser.add_argument('output_massfun_file')
	parser.add_argument('output_halobias_file')

	args = parser.parse_args()
		
	
	## read cosmological params, compute linear Pk with CAMB
	
	header_file = args.header_file

	cf = config.AbacusConfigFile(header_file)
	omega_m = cf.Omega_M	# at z=0
	redshift = cf.redshift
	sigma_8 = cf.sigma_8
	ns = cf.ns
	ombh2 = cf.ombh2
	omch2 = cf.omch2
	w0 = cf.w0
	H0 = cf.H0


	## compute (linear) power spectrum

	k, P = compute_hod.eisenstein_hu_pk(ombh2=ombh2, omch2=omch2, H0=H0, ns=ns, w0=w0,
										sigma8=sigma_8, redshift=redshift)


	## convenience functions for mass function fitting formulae

	dndm = lambda M: compute_hod.dndm_tinker(M, z=redshift, k=k, P=P, Omega_M=omega_m)
	bias = lambda M: compute_hod.compute_linear_bias(M, k, P, omega_m=omega_m)
	
	dndm_vec = np.vectorize(dndm)
	bias_vec = np.vectorize(bias)

	mass_tab = np.logspace(10., 16., 512)
	massfun_tab = dndm_vec(mass_tab)
	bias_tab = bias_vec(mass_tab)


	## save outputs
	
	np.savetxt(args.output_pk_file, np.c_[k, P])
	np.savetxt(args.output_massfun_file, np.c_[mass_tab, massfun_tab])
	np.savetxt(args.output_halobias_file, np.c_[mass_tab, bias_tab])