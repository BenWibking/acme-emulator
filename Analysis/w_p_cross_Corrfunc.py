from __future__ import print_function
import argparse
import h5py as h5
import os.path as path
import numpy as np
import Corrfunc
from Corrfunc._countpairs import countpairs_rp_pi as wp

parser = argparse.ArgumentParser()

parser.add_argument('boxsize')
parser.add_argument('pimax')
parser.add_argument('bin_file')
parser.add_argument('mock_file1')
parser.add_argument('mock_file2')
parser.add_argument('output_file')

args = parser.parse_args()

boxsize = float(args.boxsize)
pimax = float(args.pimax)
nthreads = 4
binfile = path.abspath(str(args.bin_file))

infile1 = h5.File(str(args.mock_file1), 'r')
infile2 = h5.File(str(args.mock_file2), 'r')

x1 = infile1['particles']['x']
y1 = infile1['particles']['y']
z1 = infile1['particles']['z']

x2 = infile2['particles']['x']
y2 = infile2['particles']['y']
z2 = infile2['particles']['z']

xrand = boxsize*np.random.rand(int(1e7))
yrand = boxsize*np.random.rand(int(1e7))
zrand = boxsize*np.random.rand(int(1e7))

infile1.close()
infile2.close()

wp_pre_results = wp(0, nthreads, pimax, binfile, x1, y1, z1, x2, y2, z2)
RR_pre_results = wp(1, nthreads, pimax, binfile, xrand, yrand, zrand, xrand, yrand, zrand)

wp_results = []
RR_results = []

for i in range(0, int(len(wp_pre_results)/int(pimax))):
  npairs = 0
  RRpairs = 0
  for j in range(0, int(pimax)):
    index = i*int(pimax) + j
    bin_low = wp_pre_results[index][0]
    bin_high = wp_pre_results[index][1]
    npairs += wp_pre_results[index][4]
    RRpairs += RR_pre_results[index][4]
  RR_results.append((bin_low, bin_high, 0, RRpairs))
  wp_results.append((bin_low, bin_high, 0, npairs))

for i in range(0, len(wp_results)):
  RR = (len(x1)*len(x2)/1e14)*RR_results[i][3]
  wp_results[i] = wp_results[i] + ((wp_results[i][3]/RR - 1),)

'''
for i in range(0, len(wp_results)):
  RR = len(x1)*len(x2)*np.pi*pimax*(wp_results[i][1]**2 - wp_results[i][0]**2) / boxsize**3
  wp_results[i] = wp_results[i] + ((wp_results[i][3]/RR - 1),)
'''
print(wp_results)

outfile = open(str(args.output_file), 'w')

outfile.write("# rmin rmax rpavg wp npairs\n")

for bin in wp_results:
  outfile.write(str(bin[0])+" "+str(bin[1])+" "+str(bin[2])+" "+str(bin[4])+" "+str(bin[3])+"\n")

outfile.close()
