#!/usr/bin/env python

import numpy as np
import argparse
import sys
import configparser

def load_covariance_matrix(filename):
        cov = np.loadtxt(filename,unpack=False, ndmin=2)
        return cov

def load_bins_file(filename):
        bins, signal = np.loadtxt(filename,unpack=True,ndmin=2)
        return bins

def load_correlation_file(filename):
        table = np.loadtxt(filename,unpack=False, ndmin=2)
        binmin, binmax, counts, deriv = [table[:,i] for i in range(4)]   
        return deriv

def compute_fisher_matrix(output_file, input_cov_file, input_files,
                          bins_file=None, parameter_file=None, observable=None):
        """compute Fisher matrix"""
        binmask = []
        do_mask = False
        if parameter_file is not None and bins_file is not None:
                # load parameter file
                myparser = configparser.ConfigParser()
                myparser.read(parameter_file)
                params = myparser['params']
                rp_min = 0.
                rp_max = np.inf
                min_var = []
                max_var = []
                if observable == 'wp':
                        min_var = 'wp_rp_min'
                        max_var = 'wp_rp_max'
                if observable == 'DS':
                        min_var = 'DS_rp_min'
                        max_var = 'DS_rp_max'
                if min_var in params:
                        rp_min = float(params[min_var])
                if max_var in params:
                        rp_max = float(params[max_var])
                # load bins file
                bins = load_bins_file(bins_file)
                # construct binmask
                binmask = np.logical_and(bins > rp_min, bins < rp_max)
                print(binmask,file=sys.stderr)
                do_mask = True

        # load data and assemble vectors
        deriv_vectors = []
        nobs = []
        for deriv_file, param in input_files:
                deriv = load_correlation_file(deriv_file)
                if do_mask:
                        deriv = deriv[binmask]
                nobs = deriv.shape[0]
                deriv_vector = (deriv, param)
                deriv_vectors.append(deriv_vector)        
        nparams = len(deriv_vectors)

        # initialize covariance matrix (of observations)
        C = load_covariance_matrix(input_cov_file)
        all_scales_diag_sum = np.sum(np.diag(C)**(-2.0))
        if do_mask:
                C = C[binmask,:][:,binmask]

        C_inv = np.linalg.inv(C)
        print('\tobservable covariance condition number:', file=sys.stderr)
        print('\t'+str(np.linalg.cond(C_inv)), file=sys.stderr)

        # initialize Fisher matrix
        F = np.zeros((nparams,nparams))

        # loop over Fisher matrix elements and compute dot product
        for i in range(nparams):
                for j in range(nparams):
                        deriv_i, param_i = deriv_vectors[i]
                        deriv_j, param_j = deriv_vectors[j]
                        F[i,j] = np.matmul(deriv_i, np.matmul(C_inv, deriv_j))

        # print eigenvalues
        #w, v = np.linalg.eig(F)
        #print(w,file=sys.stderr)

        # check condition number -- if large (>1e6), this indicates something went wrong
        condition = np.linalg.cond(F)
        if condition>1e6:
                print("",file=sys.stderr)
                print("WARNING: bad condition number! these results are probably wrong!",file=sys.stderr)
                print("condition number of Fisher matrix:", condition, file=sys.stderr)
                print("",file=sys.stderr)    

        # output Fisher matrix
        param_names = [p for x,p in deriv_vectors]
        header_string = ' '.join(param_names)
        np.savetxt(output_file, F, delimiter='\t',header=header_string)


parser = argparse.ArgumentParser()
parser.add_argument('output_file',help='txt file output for Fisher matrix')
parser.add_argument('covariance_matrix_file',help='covariance matrix for observable')
parser.add_argument('--bins_file',default=None,help='spatial bins file')
parser.add_argument('--parameter_file',default=None,help='parameter file')
parser.add_argument('--observable',choices=['wp','DS'],help='observable we are computing')
parser.add_argument('-f','--input_file',nargs=2,action='append',help='derivative files for a given parameter')
# this returns a list of tuples, one item for each input file
# -- the first part of the tuple should be the autocorrelation file
# -- the second part of the tuple should be the name of the parameter (to match up with the covariance matrix)

args = parser.parse_args()

compute_fisher_matrix(args.output_file,
                      args.covariance_matrix_file,
                      args.input_file,
                      bins_file=args.bins_file,
                      parameter_file=args.parameter_file,
                      observable=args.observable)
