#!/usr/bin/env python

import argparse
import numpy as np
import scipy.optimize
import h5py as h5
import configparser

from compute_mock import compute_ngal, compute_HOD_parameters


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	
	parser.add_argument('hod_params_path')
	parser.add_argument('header_path')
	parser.add_argument('halo_path')
	parser.add_argument('mass_fun_path')
	parser.add_argument('output_file')

	args = parser.parse_args()
	
	
	## read meta-HOD parameters
	
	myconfigparser = configparser.ConfigParser()
	myconfigparser.read(args.hod_params_path)
	params = myconfigparser['params']
	

	## find HOD parameters
	
	logMmin, logM0, logM1 = compute_HOD_parameters(ngal=float(params['ngal']),
	                                                   siglogM=float(params['siglogm']),
	                                                   M0_over_M1=float(params['m0_over_m1']),
	                                                   M1_over_Mmin=float(params['m1_over_mmin']),
	                                                   alpha=float(params['alpha']),
	                                                   f_cen=float(params['f_cen']),
	                                                   halos=args.halo_path,
	                                                   header=args.header_path,
	                                                   mass_fun_file=args.mass_fun_path)

	np.savetxt(args.output_file, np.array(logMmin, ndmin=1))
