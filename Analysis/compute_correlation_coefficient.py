import numpy as np
import argparse
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('output_file',help='galaxy-matter (pseudo)correlation coefficient output file')
parser.add_argument('galaxy_galaxy_file',help='galaxy-galaxy correlation function')
parser.add_argument('matter_matter_file',help='matter-matter correlation function')
parser.add_argument('galaxy_matter_file',help='galaxy-matter correlation function')

args = parser.parse_args()
                    
def load_correlation_file(filename):
    table = np.loadtxt(filename,unpack=False)
    binmin, binmax, counts, corr = [table[:,i] for i in range(4)]                        
    return binmin,binmax,corr

def compute_correlation_coefficient(gg_file,mm_file,gm_file,output_file):
    gg_binmin, gg_binmax, gg_corr = load_correlation_file(gg_file)
    mm_binmin, mm_binmax, mm_corr = load_correlation_file(mm_file)
    gm_binmin, gm_binmax, gm_corr = load_correlation_file(gm_file)

    if np.array_equal(gg_binmin,mm_binmin) and np.array_equal(gg_binmin,gm_binmin) and np.array_equal(gg_binmax,mm_binmax) and np.array_equal(gg_binmax,gm_binmax):
        binmin = gg_binmin
        binmax = gg_binmax

        corr_coef_sq = gm_corr**2 / (gg_corr*mm_corr)

        # TODO: apply large-scale asymptotic regularization
        # (i.e. at large scales, r -> 1, so r_sq -> 1)
        #corr_coef_sq[binmax >= 50.] = 1.0

        # compute square root
        from numpy.lib.scimath import sqrt 
        corr_coef = sqrt(corr_coef_sq)

        np.savetxt(output_file, np.c_[binmin, binmax, np.zeros(corr_coef.shape[0]), corr_coef],
                       delimiter='\t')
    else:
        print("bins do not match!")
        print(np.array_equal(gg_binmin,mm_binmin))
        print(np.array_equal(gg_binmin,gm_binmin))
        print(gg_binmin,gm_binmin)
        print(np.array_equal(gg_binmax,mm_binmax))
        print(np.array_equal(gg_binmax,gm_binmax))
        sys.exit(1)


if __name__ == "__main__":
    bins = compute_correlation_coefficient(args.galaxy_galaxy_file, args.matter_matter_file, args.galaxy_matter_file, args.output_file)

