from __future__ import print_function
import argparse
import h5py as h5
import os.path as path
import numpy as np
import Corrfunc
from Corrfunc._countpairs import countpairs_wp as wp

parser = argparse.ArgumentParser()

parser.add_argument('boxsize')
parser.add_argument('pimax')
parser.add_argument('bin_file')
parser.add_argument('HOD_mock_file')
parser.add_argument('output_file')

args = parser.parse_args()

boxsize = float(args.boxsize)
pimax = float(args.pimax)
nthreads = 4
binfile = path.abspath(str(args.bin_file))

infile = h5.File(str(args.HOD_mock_file), 'r')

x = infile['particles']['x']
y = infile['particles']['y']
z = infile['particles']['z']

infile.close()

wp_results = wp(boxsize, pimax, nthreads, binfile, x, y, z)

#print(wp_results)

outfile = open(str(args.output_file), 'w')

outfile.write("# rmin rmax rpavg wp npairs\n")

for bin in wp_results:
  outfile.write(str(bin[0])+" "+str(bin[1])+" "+str(bin[2])+" "+str(bin[3])+" "+str(bin[4])+"\n")

outfile.close()
