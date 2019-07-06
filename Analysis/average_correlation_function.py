#!/usr/bin/env python

import numpy as np
import argparse


def average_functions(input_files, output_file):

	binmin0, binmax0, _, corr0 = np.loadtxt(input_files[0], unpack=True)
	corrs = np.zeros((len(input_files), corr0.shape[0]))
	corrs[0, :] = corr0
	
	for i, f in enumerate(input_files[1:]):
		binmin, binmax, _, corrs[i, :] = np.loadtxt(f, unpack=True)
		assert(np.allclose(binmin, binmin0))
		assert(np.allclose(binmax, binmax0))

	# estimate the sample mean
	avg_corr = np.mean(corrs, axis=0)
	
	# estimate the error on the mean
	err_corr = np.std(corrs, axis=0) / np.sqrt( float(len(input_files)) )

	np.savetxt(output_file, np.c_[binmin0, binmax0, err_corr, avg_corr], delimiter='\t')


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	
	parser.add_argument('output_file',help='averaged correlation function output')
	parser.add_argument('input_files',nargs='*',help='input correlation functions')
	
	args = parser.parse_args()
	
	bins = average_functions(args.input_files, args.output_file)

