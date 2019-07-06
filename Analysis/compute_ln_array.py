import numpy as np
import argparse
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('input_file',help='input file')
parser.add_argument('output_file',help='output file')

args = parser.parse_args()
                    
def compute_ln(input_file,output_file):
    arr = np.loadtxt(input_file, ndmin=2)

    ## compute logarithm
    ln_arr = np.log(arr)

    np.savetxt(output_file, ln_arr, delimiter='\t')

if __name__ == "__main__":
    bins = compute_ln(args.input_file, args.output_file)

