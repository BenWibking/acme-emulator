import numpy as np
import argparse
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('input_file',help='input file')
parser.add_argument('output_file',help='output file')

args = parser.parse_args()

def load_correlation_file(filename):
    table = np.loadtxt(filename,unpack=False)
    binmin, binmax, counts, corr = [table[:,i] for i in range(4)]                        
    return binmin,binmax,corr

def compute_exp(input_file,output_file):
    binmin, binmax, corr = load_correlation_file(input_file)

    ## compute exponent
    exp_corr = np.exp(corr)

    np.savetxt(output_file, np.c_[binmin, binmax, np.zeros(exp_corr.shape[0]), exp_corr],
               delimiter='\t')

if __name__ == "__main__":
    bins = compute_exp(args.input_file, args.output_file)

