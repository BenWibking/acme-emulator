import argparse
import h5py as h5
import os.path as path
import numpy as np
import Corrfunc
from Corrfunc.theory.xi import xi as corrfunc_xi

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	parser.add_argument('boxsize')
	parser.add_argument('bin_file')
	parser.add_argument('HOD_mock_file')
	parser.add_argument('output_file')
	parser.add_argument('--ignore_weights',default=False,action='store_true')
	parser.add_argument('--centrals_only',default=False,action='store_true')
	parser.add_argument('--do-not-subsample',default=True,action='store_true')
	
	args = parser.parse_args()
	
	boxsize = float(args.boxsize)
	nthreads = 8
	binfile = path.abspath(str(args.bin_file))
	
	infile = h5.File(str(args.HOD_mock_file), 'r')
	
	x = infile['particles']['x']
	y = infile['particles']['y']
	z = infile['particles']['z']
	if not args.ignore_weights:
			weight = infile['particles']['weight']
	else:
			weight = None
	
	if args.centrals_only == True:
			is_sat = infile['particles']['is_sat']
			x = x[is_sat == 0]
			y = y[is_sat == 0]
			z = z[is_sat == 0]
			weight = weight[is_sat == 0]
	
	infile.close()
	
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
	
	x = convert_to_native_endian(x)
	y = convert_to_native_endian(y)
	z = convert_to_native_endian(z)
	if not args.ignore_weights:
			weight = convert_to_native_endian(weight)
	
	# subsample particle arrays to limit to 10^6 points
	# we seem to hit cache inefficiencies above 1e6, so we subsample and average
	# (reason: 32K L1 cache means that exactly 4 * 1e3 doubles will fit, average no. per grid cell)
	npart = len(x)
	npart_target = int(1.0e6)
	
	def compute_xi(x,y,z,w=None):
			if w is not None:
				xi_results = corrfunc_xi(boxsize,nthreads,binfile,
										 x.astype(np.float64),
										 y.astype(np.float64),
										 z.astype(np.float64),
										 weights=w.astype(np.float64),
										 weight_type='pair_product')
			else:
				xi_results = corrfunc_xi(boxsize,nthreads,binfile,x,y,z)

			rmin = xi_results['rmin']
			rmax = xi_results['rmax']
			ravg = xi_results['ravg']
			xi = xi_results['xi']
			npairs = xi_results['npairs']
			weightavg = xi_results['weightavg']
			return rmin,rmax,npairs,xi
	
	xi = []
	npairs = []
	
	do_subsample = not args.do_not_subsample
	if npart > npart_target and do_subsample == True:
			print('doing weighted subsample (Np: {} Np_subsample: {} subsample_fraction: {})'.format(npart,npart_target,npart_target/npart),file=sys.stderr,flush=True)
	
			nsubsamples = 20
			subsample_weight = 1.0/float(nsubsamples)
			rng_seed = 42
			rng = np.random.RandomState(rng_seed)
			if weight is not None:
				weight_normalized = weight / np.sum(weight)
			else:
				weight_normalized = None
	
			for isubsample in range(nsubsamples):
				idx = rng.choice(npart, size=npart_target, replace=False, p=weight_normalized)
				x_sub = x[idx]
				y_sub = y[idx]
				z_sub = z[idx]
	
				if xi == []:
					rmin,rmax,npairs,xi = compute_xi(x_sub,y_sub,z_sub)
					npairs = npairs.astype(np.float64) * subsample_weight
					xi *= subsample_weight
				else:
					this_rmin,this_rmax,this_pairs,this_xi = compute_xi(x_sub,y_sub,z_sub)
					xi += subsample_weight * this_xi
					npairs += subsample_weight * this_pairs
	
	else:
			rmin,rmax,npairs,xi = compute_xi(x,y,z,w=weight)
	
	outfile = open(str(args.output_file), 'w')
	
	outfile.write("# rmin rmax npairs xi\n")
	
	for i in range(xi.shape[0]):
			outfile.write(str(rmin[i])+" "+str(rmax[i])+" "+str(npairs[i])+" "+str(xi[i])+"\n")
	
	outfile.close()
