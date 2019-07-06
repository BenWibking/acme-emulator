import numpy as np
import argparse
from emulator import *

def emulate_files(fiducial_file, deriv_files, output_file):
        # convert arguments
        deriv_filenames, fiducial_params, params = zip(*deriv_files)
        fiducial_params = np.array([float(x) for x in fiducial_params])
        params = np.array([float(x) for x in params])

        # read emulator data
        data = emulator_data_from_files(fiducial_file,deriv_filenames)
        binmin,binmax,fiducial_model,derivs = data

        # emulate
        model = emulate(fiducial_model, derivs, params, fiducial_params)

        # save model output
        np.savetxt(output_file, np.c_[binmin, binmax, np.zeros(model.shape[0]), model], delimiter='\t')

parser = argparse.ArgumentParser()
parser.add_argument('output_file',help='txt file output for emulated observable')
parser.add_argument('fiducial_file',help='txt file output for fiducial emulated observable')
parser.add_argument('-f','--input_file',nargs=3,action='append',help='derivative file')
# this returns a list of tuples, one item for each input file
# -- the first part of the tuple is the filename
# -- the second part of the tuple is the parameter value of the fiducial model
# -- the third part of the tuple is the parameter value we wish to emulate

args = parser.parse_args()

emulate_files(args.fiducial_file, args.input_file, args.output_file)
