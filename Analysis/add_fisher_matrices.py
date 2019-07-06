import numpy as np
import argparse
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('output_fisher_file',help='output Fisher matrix file')
parser.add_argument('output_cov_file',help='output parameter covariance file')
parser.add_argument('fisher_matrices',nargs='*',help='Fisher matrix files')

args = parser.parse_args()
                    
def load_matrix_file(filename):
    with open(filename) as f:
        header = f.readline()
    matrix = np.genfromtxt(filename,unpack=False)
    return header,matrix

def add_Fisher_matrices(output_fisher_file, output_cov_file, fisher_matrix_files):
    # read data
    header,fisher = load_matrix_file(fisher_matrix_files[0])
    for matrix_file in fisher_matrix_files[1:]:
        this_header, this_fisher = load_matrix_file(matrix_file)
        if this_header == header:
            fisher += this_fisher
        else:
            raise Exception("headers do not match!")

    if fisher.shape is ():
        cov = 1.0/fisher
    else:
        cov = np.linalg.inv(fisher)

    # save
    header = header.replace("# ","")
    header = header.strip()

    if fisher.shape is ():
        with open(output_fisher_file,'w') as f:
            f.write('#')
            f.write(header)
            f.write('\n')
            f.write(str(fisher))
        with open(output_cov_file,'w') as f:
            f.write('#')
            f.write(header)
            f.write('\n')
            f.write(str(cov))
    else:
        np.savetxt(output_fisher_file, fisher, delimiter='\t', header=header)
        np.savetxt(output_cov_file, cov, delimiter='\t', header=header)

if __name__ == "__main__":
    bins = add_Fisher_matrices(args.output_fisher_file, args.output_cov_file, args.fisher_matrices)

