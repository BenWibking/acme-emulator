#!/usr/bin/env python
from os import path
import argparse
import subprocess
import numpy as np
import h5py as h5

import config

def compute(mock_filename,header_filename,output_filename,rmin=0.1,rmax=125.0,nbins=40,njackknife=8):
    cf = config.AbacusConfigFile(header_filename)
    boxsize = cf.boxSize
    omeganow_m = cf.OmegaNow_m
    omega_m = cf.Omega_M
    redshift = cf.redshift

    # ./auto nbins rmin rmax box_size njackknife_samples filename
    ## compute galaxy autocorrelation - modify output so header is a comment line
    ## --modify output so that pair counts include double counts
    with open(output_filename, 'w') as output_file:
        script_path = path.dirname(path.abspath(__file__))+"/fastcorrelation/auto"
        call_string = [script_path,str(nbins),str(rmin),str(rmax),str(boxsize),str(njackknife),mock_filename]
        subprocess.call(call_string,stdout=output_file)

parser = argparse.ArgumentParser()
parser.add_argument('HOD_mock_file')
parser.add_argument('header_file')
parser.add_argument('output_file')
#parser.add_argument('mindist',type=float)
#parser.add_argument('maxdist',type=float)
#parser.add_argument('nbins',type=int)

args = parser.parse_args()

compute(args.HOD_mock_file,args.header_file,args.output_file)
        #rmin=args.mindist,rmax=args.maxdist,nbins=args.nbins)
