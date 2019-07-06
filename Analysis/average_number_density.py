import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('output_file',help='averaged correlation function output')
parser.add_argument('input_files',nargs='*',help='input correlation functions to be averaged')

args = parser.parse_args()
                    
def average_functions(input_files,output_file):
    table = np.loadtxt(input_files[0])
    for f in input_files[1:]:
        table += np.loadtxt(f)

    table /= float(len(input_files))
    np.savetxt(output_file, table, delimiter='\t')

if __name__ == "__main__":
    bins = average_functions(args.input_files, args.output_file)

