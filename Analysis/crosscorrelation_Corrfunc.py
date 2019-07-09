from __future__ import print_function
import argparse
import h5py as h5
import config
import os.path as path
import os
import numpy as np
from Corrfunc.theory.DD import DD as compute_DD

# fix endianness issues
import sys
def convert_to_native_endian(array):
  system_is_little_endian = (sys.byteorder == 'little')
  array_is_little_endian = (array.dtype.byteorder == '<')
  is_native_endian = (system_is_little_endian and array_is_little_endian) or (not system_is_little_endian and not array_is_little_endian) or (array.dtype.byteorder == '=')
  if not is_native_endian:
    return array.byteswap().newbyteorder()
  else:
    return array

parser = argparse.ArgumentParser()

parser.add_argument('boxsize')
parser.add_argument('bin_file')
parser.add_argument('mock_file_1')
parser.add_argument('mock_file_2')
parser.add_argument('output_file')
parser.add_argument('--centrals_only',default=False,action='store_true')
parser.add_argument('--do-not-subsample',default=True,action='store_true')

args = parser.parse_args()

if 'OMP_NUM_THREADS' not in os.environ.keys():
	nthreads = 2
else:
	nthreads = int(os.environ['OMP_NUM_THREADS'])
	
	
boxsize = float(args.boxsize)	
binfile = path.abspath(str(args.bin_file))

infile1 = h5.File(str(args.mock_file_1), 'r')

x1 = convert_to_native_endian(infile1['particles']['x'])
y1 = convert_to_native_endian(infile1['particles']['y'])
z1 = convert_to_native_endian(infile1['particles']['z'])
if 'weight' in infile1['particles'].dtype.fields:
  w1 = convert_to_native_endian(infile1['particles']['weight'])
else:
  w1 = None

if args.centrals_only == True and '/particles/is_sat' in infile1:
  is_sat1 = infile1['particles']['is_sat']
  x1 = x1[is_sat1 == 0]
  y1 = y1[is_sat1 == 0]
  z1 = z1[is_sat1 == 0]
  w1 = w1[is_sat1 == 0]

infile1.close()

infile2 = h5.File(str(args.mock_file_2), 'r')

x2 = convert_to_native_endian(infile2['particles']['x'])
y2 = convert_to_native_endian(infile2['particles']['y'])
z2 = convert_to_native_endian(infile2['particles']['z'])
if 'weight' in infile2['particles'].dtype.fields:
  w2 = convert_to_native_endian(infile2['particles']['weight'])
else:
  w2 = None

if args.centrals_only == True and '/particles/is_sat' in infile2:
  is_sat2 = infile2['particles']['is_sat']
  x2 = x2[is_sat2 == 0]
  y2 = y2[is_sat2 == 0]
  z2 = z2[is_sat2 == 0]
  w2 = w2[is_sat2 == 0]

infile2.close()

def compute_xi(x1,y1,z1,x2,y2,z2,w1=None,w2=None):
  # compute DD, RR
  # results_DD = [[rmin,rmax,ravg,npairs,weightavg],...]
  # (npairs is weighted by weights1,weights2)
  if w1 is not None or w2 is not None:
    if w1 is None:
      w1 = convert_to_native_endian(np.ones(x1.shape,dtype=x1.dtype))    
    elif w2 is None:
      w2 = convert_to_native_endian(np.ones(x2.shape,dtype=x2.dtype))

    results_DD = compute_DD(0, nthreads, binfile,
                    x1.astype(np.float64),
                    y1.astype(np.float64),
                    z1.astype(np.float64),
                    weights1=w1.astype(np.float64),
                    X2=x2.astype(np.float64),
                    Y2=y2.astype(np.float64),
                    Z2=z2.astype(np.float64),
                    weights2=w2.astype(np.float64),
                    weight_type='pair_product')

    nbins = len(results_DD)
    rmin = np.zeros(nbins)
    rmax = np.zeros(nbins)
    DD = np.zeros(nbins)
    RR = np.zeros(nbins)
    for i in range(nbins):
      rmin[i] = results_DD[i][0]
      rmax[i] = results_DD[i][1]
      DD[i] = results_DD[i][3] * results_DD[i][4] # need to multiply pair counts by weights
      RR[i] = np.sum(w1)*np.sum(w2)*(4./3.)*np.pi*(rmax[i]**3 - rmin[i]**3) / boxsize**3

  else: # no weights provided for either set, just count pairs
    results_DD = compute_DD(0, nthreads, binfile,x1,y1,z1,X2=x2,Y2=y2,Z2=z2)

    nbins = len(results_DD)
    rmin = np.zeros(nbins)
    rmax = np.zeros(nbins)
    DD = np.zeros(nbins,dtype=np.uint64)
    RR = np.zeros(nbins)
    for i in range(nbins):
      rmin[i] = results_DD[i][0]
      rmax[i] = results_DD[i][1]
      DD[i] = results_DD[i][3]
      RR[i] = float(x1.shape[0])*float(x2.shape[0])*(4./3.)*np.pi*(rmax[i]**3 - rmin[i]**3) / boxsize**3

  # compute xi from DD and RR
  xi = DD/RR - 1.0
  return rmin,rmax,DD,xi

rmin = []
rmax = []
DD = []
xi = []

# subsample particle arrays to limit to 10^6 points
# we seem to hit cache inefficiencies above 1e6, so we subsample and average
# (reason: 32K L1 cache means that exactly 4 * 1e3 doubles will fit, average no. per grid cell)
npart_target = int(1.0e6)

do_subsample = not args.do_not_subsample
if do_subsample:
  # figure out which input file corresponds to galaxies, which are matter
  if w1 is not None and w2 is None:
    npart_gal = len(x1)
    x_gal = x1
    y_gal = y1
    z_gal = z1
    w_gal = w1
    x_dm = x2
    y_dm = y2
    z_dm = z2
    w_dm = w2
  elif w2 is not None and w1 is None:
    npart_gal = len(x2)
    x_gal = x2
    y_gal = y2
    z_gal = z2
    w_gal = w2
    x_dm = x1
    y_dm = y1
    z_dm = z1
    w_dm = w1
  else:
    print("only one of the input files should have weights! exiting.")
    exit(1)  

if do_subsample:
  if npart_gal > npart_target:
    print('doing weighted subsample (Np: {} Np_subsample: {} subsample_fraction: {})'.format(npart_gal,npart_target,npart_target/npart_gal),file=sys.stderr,flush=True)

    nsubsamples = 20
    subsample_weight = 1.0/float(nsubsamples)
    rng_seed = 42
    rng = np.random.RandomState(rng_seed)

    w_gal_normalized = w_gal / np.sum(w_gal)
    for isubsample in range(nsubsamples):
      idx_gal = rng.choice(npart_gal, size=npart_target, replace=False, p=w_gal_normalized)
      x_gal_sub = x_gal[idx_gal]
      y_gal_sub = y_gal[idx_gal]
      z_gal_sub = z_gal[idx_gal]

      if xi == []:
        rmin,rmax,DD,xi = compute_xi(x_gal_sub,y_gal_sub,z_gal_sub,x_dm,y_dm,z_dm)
        DD = DD.astype(np.float64) * subsample_weight
        xi *= subsample_weight
      else:
        this_rmin,this_rmax,this_DD,this_xi = compute_xi(x_gal_sub,y_gal_sub,z_gal_sub,
                                                       x_dm,y_dm,z_dm)
        xi += subsample_weight * this_xi
        DD += subsample_weight * this_DD
else:
  rmin,rmax,DD,xi = compute_xi(x1,y1,z1,x2,y2,z2,w1=w1,w2=w2)


## write output
print("output to: {}".format(args.output_file),file=sys.stderr)
with open(str(args.output_file), 'w') as outfile:
  outfile.write("# rmin rmax npairs xi\n")
  for i in range(xi.shape[0]):
    outfile.write("{} {} {} {}\n".format(rmin[i],rmax[i],DD[i],xi[i]))
