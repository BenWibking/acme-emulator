#!/usr/bin/env python

import numpy as np
import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('fiducial_ratio_file')
	parser.add_argument('output_ratio_correction')
	parser.add_argument('ratio_phases_files', nargs='*')
	parser.add_argument('--multiplicative', default=False, action='store_true',
						help='compute multiplicative correction')

	args = parser.parse_args()
	
	
	## read fiducial ratio
	
	wpmin, wpmax, _, wpratio_fid = np.loadtxt(args.fiducial_ratio_file, unpack=True)
	
	
	## compute mean ratio from phases
	
	wpratio_mean = np.zeros_like(wpratio_fid)
	Nphases = len(args.ratio_phases_files)
	print(f"Number of phases: {Nphases}")
	
	for rfile in args.ratio_phases_files:
		t_wpmin, t_wpmax, _, t_wpratio = np.loadtxt(rfile, unpack=True)
		assert( np.all(t_wpmin == wpmin) )
		assert( np.all(t_wpmax == wpmax) )
		wpratio_mean += t_wpratio
		
	wpratio_mean *= (1.0/Nphases)
	
	
	## compute correction term
	
	if args.multiplicative == True:
		wpratio_correction = wpratio_mean / wpratio_fid
	else:
		wpratio_correction = wpratio_mean - wpratio_fid

	print(f"correction term = {wpratio_correction}")
	
	
	## save to file
	
	np.savetxt(args.output_ratio_correction,
				np.c_[wpmin, wpmax, np.zeros_like(wpratio_correction), wpratio_correction])
	