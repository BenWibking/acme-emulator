import numpy as np
import argparse
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('output_file',help='galaxy bias output file')
parser.add_argument('galaxy_bias_file',help='(scale-dependent) galaxy bias')
parser.add_argument('matter_bias_file',help='(scale-dependent) nonlinear matter "bias"')
parser.add_argument('linear_matter_file',help='linear matter correlation function')

args = parser.parse_args()
                    
def load_correlation_file(filename):
    table = np.loadtxt(filename,unpack=False,dtype=np.complex128)
    binmin, binmax, counts, corr = [table[:,i] for i in range(4)]                        
    return binmin,binmax,corr

def reconstruct_xi_gg(bg_file, bm_file, mm_file, output_file):
    bg_binmin, bg_binmax, bg = load_correlation_file(bg_file)
    bm_binmin, bm_binmax, bm = load_correlation_file(bm_file)
    mm_binmin, mm_binmax, mm_corr = load_correlation_file(mm_file)

    tol=1.0e-5
    if ((bg_binmin-bm_binmin)<tol).all() and ((bg_binmax-bm_binmax)<tol).all() and \
       ((bg_binmin-mm_binmin)<tol).all() and ((bg_binmax-mm_binmax)<tol).all():
        binmin = mm_binmin
        binmax = mm_binmax

        xi_gg = ((bg*bg) * (bm*bm) * mm_corr)

        np.savetxt(output_file, np.c_[binmin.real, binmax.real, np.zeros(xi_gg.shape[0]), xi_gg.real],
                       delimiter='\t')
    else:
        print("bins do not match!")
        sys.exit(1)


if __name__ == "__main__":
    bins = reconstruct_xi_gg(args.galaxy_bias_file, args.matter_bias_file, args.linear_matter_file,
                             args.output_file)

