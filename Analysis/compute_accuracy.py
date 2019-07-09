import numpy as np
import argparse
from pathlib import Path

def load_correlation_file(filename):
    table = np.loadtxt(filename,unpack=False)
    binmin, binmax, counts, corr = [table[:,i] for i in range(4)]                        
    return binmin,binmax,corr

def compute_accuracy(true_file, emulated_file, output_file):
    import sys

    tru_binmin, tru_binmax, tru_corr = load_correlation_file(true_file)
    emu_binmin, emu_binmax, emu_corr = load_correlation_file(emulated_file)

    tol=1.0e-5
    if ((tru_binmin-emu_binmin)<tol).all() and ((tru_binmax-emu_binmax)<tol).all():
        binmin = tru_binmin
        binmax = tru_binmax

        frac_accuracy = (emu_corr - tru_corr) / tru_corr

        np.savetxt(output_file, np.c_[binmin, binmax, np.zeros(frac_accuracy.shape[0]), frac_accuracy],
                       delimiter='\t')
    else:
        print("bins do not match!")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('true_file')
    parser.add_argument('emulated_file')
    parser.add_argument('output_file',help='fractional accuracy output file')
    
    args = parser.parse_args()            

    compute_accuracy(args.true_file, args.emulated_file, args.output_file)

