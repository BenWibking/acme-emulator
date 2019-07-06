import numpy as np
import argparse
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('galaxy_galaxy_file',help='correlation function')
parser.add_argument('output_file',help='galaxy bias output file')

args = parser.parse_args()

def load_correlation_file(filename):
    table = np.loadtxt(filename,unpack=False)
    binmin, binmax, counts, corr = [table[:,i] for i in range(4)]
    return binmin,binmax,corr

def smooth(in_binmin,in_binmax,corr):
    left = corr[0:][::2]
    right = corr[1:][::2]
    smooth_corr = 0.5*(left + right)
    binmin = in_binmin[0:][::2]
    binmax = in_binmax[1:][::2]
    return binmin, binmax, smooth_corr

def smooth_function(input_file,output_file):
    in_binmin, in_binmax, corr = load_correlation_file(input_file)

    binmin, binmax, smooth_corr = smooth(in_binmin,in_binmax,corr)
    binmin, binmax, smooth_corr = smooth(binmin,binmax,smooth_corr)

    np.savetxt(output_file, np.c_[binmin, binmax, np.zeros(smooth_corr.shape[0]), smooth_corr],
                delimiter='\t')

if __name__ == "__main__":
    smooth_function(args.galaxy_galaxy_file, args.output_file)
