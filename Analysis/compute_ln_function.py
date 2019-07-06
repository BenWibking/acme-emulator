import numpy as np
import argparse
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('input_file',help='input file')
parser.add_argument('output_file',help='output file')

args = parser.parse_args()
                    
def load_correlation_file(filename):
    table = np.loadtxt(filename,unpack=False,dtype=np.complex128)
    binmin, binmax, counts, corr = [table[:,i].real for i in range(4)]
    return binmin,binmax,corr

def compute_ln(input_file,output_file):
    binmin, binmax, corr = load_correlation_file(input_file)

    ## compute logarithm
    ln_corr = np.log(corr)

    np.savetxt(output_file, np.c_[binmin, binmax, np.zeros(ln_corr.shape[0]), ln_corr],
               delimiter='\t')

if __name__ == "__main__":
    bins = compute_ln(args.input_file, args.output_file)

