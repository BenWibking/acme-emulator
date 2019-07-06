import numpy as np
import argparse
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('output_file',help='galaxy bias output file')
parser.add_argument('galaxy_galaxy_file',help='galaxy-galaxy correlation function')
parser.add_argument('matter_matter_file',help='matter-matter correlation function')
parser.add_argument('--regularize',default=False,action='store_true',help='average large-scales and set to constant')

args = parser.parse_args()
                    
def load_correlation_file(filename):
    table = np.loadtxt(filename,unpack=False)
    binmin, binmax, counts, corr = [table[:,i] for i in range(4)]                        
    return binmin,binmax,corr

def compute_bias(gg_file,mm_file,output_file,regularize):
    gg_binmin, gg_binmax, gg_corr = load_correlation_file(gg_file)
    mm_binmin, mm_binmax, mm_corr = load_correlation_file(mm_file)

    tol=1.0e-5
    if ((gg_binmin-mm_binmin)<tol).all() and ((gg_binmax-mm_binmax)<tol).all():
        binmin = mm_binmin
        binmax = mm_binmax

        bias_sq = (gg_corr/mm_corr)

        ## regularize on large scales
        rinner = 8.
        router = 30. # Mpc/h
        bias_sq_lin = np.mean(bias_sq[np.logical_and(binmin >= rinner, binmax <= router)])

        ## regularize
#        if regularize:
#            bias_sq[binmax >= router] = bias_sq_lin

        ## prevent NaNs
#        bias_sq[bias_sq <= 0.] = bias_sq_lin

        ## compute square root
        #from numpy.lib.scimath import sqrt 
        bias = np.sqrt(bias_sq) # returns complex numbers if needed

        np.savetxt(output_file, np.c_[binmin, binmax, np.zeros(bias.shape[0]), bias],
                       delimiter='\t')
    else:
        print("bins do not match!")
        sys.exit(1)


if __name__ == "__main__":
    bins = compute_bias(args.galaxy_galaxy_file, args.matter_matter_file, args.output_file, args.regularize)

