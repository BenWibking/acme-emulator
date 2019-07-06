import numpy as np
import scipy.interpolate
import argparse

if __name__ == '__main__':
	
	"""convert baryonic effects ratio on DeltaSigma to a new mock observation of DeltaSigma."""
	
	parser = argparse.ArgumentParser()
	parser.add_argument('input_DS')
	parser.add_argument('input_baryon_ratio')
	parser.add_argument('output_DS')
	
	args = parser.parse_args()
	
	
	## read in files
	
	rp_ratio, ds_ratio, ups_ratio1, ups_ratio2, ups1_ratio, ups2_ratio, ups3_ratio, ups4_ratio = np.loadtxt(args.input_baryon_ratio, unpack=True)
	
	rpmin, rpmax, ds_err, ds_mock = np.loadtxt(args.input_DS, unpack=True)
	rpmid = 0.5*(rpmin+rpmax)
	
	
	## bins may not match, so let's interpolate the baryonic effects ratio
	## for values *above* the given input rp, assume the effect ratio is 1 (i.e., no effect)
	
	ds_ratio_interp = scipy.interpolate.interp1d(rp_ratio, ups_ratio1,
												 kind='cubic', bounds_error=False, fill_value=(np.NaN, 1.0))

	baryonic_modification = ds_ratio_interp(rpmid)

	print(f"rp_mid = {rpmid}")
	print(f"baryonic modification: {baryonic_modification}")
	
	
	## save output
	
	np.savetxt(args.output_DS, np.c_[rpmin, rpmax, ds_err, ds_mock * baryonic_modification])