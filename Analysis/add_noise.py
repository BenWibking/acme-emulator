import numpy as np
import argparse
from pathlib import Path

def load_correlation_file(filename):
    table = np.loadtxt(filename,unpack=False)
    binmin, binmax, counts, corr = [table[:,i] for i in range(4)]                        
    return binmin,binmax,corr

def compute_noise(input_file,cov_file,output_file):
    binmin, binmax, corr = load_correlation_file(input_file)

    ## compute noise
    cov = np.loadtxt(cov_file)
    noise = np.random.multivariate_normal(np.zeros(corr.shape[0]), cov)
    obs_corr = corr + noise

    np.savetxt(output_file, np.c_[binmin, binmax, np.zeros(obs_corr.shape[0]), obs_corr],
               delimiter='\t')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file',help='input file')
    parser.add_argument('cov_file',help='covariance matrix')
    parser.add_argument('output_file',help='output file')
    
    args = parser.parse_args()

    bins = compute_noise(args.input_file, args.cov_file, args.output_file)

