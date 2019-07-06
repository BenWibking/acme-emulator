import numpy as np
import argparse
import config
from pathlib import Path
from operator import itemgetter
import sys

def second_order_finite_differences(function_value, delta_x):
        """compute three-point finite difference approximation to first derivative"""
        if len(function_value) == 3:
                return (function_value[2] - function_value[0])/(abs(delta_x[1]) + abs(delta_x[0]))
        elif len(function_value) == 2:
                return (function_value[1] - function_value[0]) / delta_x[0]
        else:
                raise Exception('wrong number of inputs for finite differences computation.')

def compute_derivative(input_files, output_file, log_parameter=False, debug=False):
        """compute partial derivative at each point of the function."""

        # convert input_files into (string, float) tuples [otherwise .sort() will sort lexicographically!]
        input_files = [(f,float(p)) for f,p in input_files]

        # sort input_files by param_value from low to high:
        input_files.sort(key=itemgetter(1), reverse=False) # itemgetter(1) == second item in tuple

        # determine array sizes
        filename_0, param_value_0 = input_files[0]
        arr = np.loadtxt(filename_0, ndmin=2)
        derivative = np.zeros(arr.shape[0])
        function_values = np.zeros((arr.shape[0],len(input_files)))
        param_values = np.zeros(len(input_files))

        # read first file
        function_values[:,0] = arr[:]
        param_values[0] = param_value_0

        # read remaining files
        offset = 1
        for i, (f, p) in enumerate(input_files[offset:]):
                arr = np.loadtxt(f, ndmin=2)
                function_values[:,offset+i] = arr[:]
                param_values[offset+i] = p                 

        if debug==True:
                print("param_values: {}".format(param_values),file=sys.stderr)

        if log_parameter is True:
                param_values = np.log(param_values)

        # compute \Delta p
        deltas_p = np.diff(param_values)
        if np.all(deltas_p > 0.):
                pass
        else:
                print(deltas_p)
                print(param_values)
                raise Exception("derivative parameters not sorted!")

        if debug == True:
                print("function_values: {}".format(function_values),file=sys.stderr)
                print("param_values: {}".format(param_values),file=sys.stderr)
                print("delta param_values: {}".format(deltas_p),file=sys.stderr)

        # compute derivative of f with finite differences
        for i in range(function_values.shape[0]):
                derivative[i] = second_order_finite_differences(function_values[i,:], deltas_p)

        if log_parameter is True:
                print("log parameter derivative",file=sys.stderr)
        else:
                print("linear parameter derivative",file=sys.stderr)
        print(derivative,file=sys.stderr)
        
        np.savetxt(output_file, np.c_[0., 0., 0., derivative], delimiter='\t')

parser = argparse.ArgumentParser()
parser.add_argument('--log_parameter',default=False,action='store_true')
parser.add_argument('output_file',help='txt file output for numerical derivative')
parser.add_argument('-f','--input_file',nargs=2,action='append',help='correlation function file')
parser.add_argument('--debug',default=False,action='store_true')
# this returns a list of tuples, one item for each input file
# -- the first part of the tuple should be the filename
# -- the second part of the tuple should be the parameter value w.r.t which we are differentiating

args = parser.parse_args()

compute_derivative(args.input_file, args.output_file, log_parameter=args.log_parameter,
                   debug=args.debug)
