import numpy as np
from pathlib import Path
import argparse

if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('input_obs_file')
	parser.add_argument('--scale-min', type=float, default=0.0)
	parser.add_argument('--scale-max', type=float, default=np.inf)
	args = parser.parse_args()


	## determine output filenames automatically

	output_ds = str( Path(args.input_obs_file).with_suffix('.ds.txt') )
	output_wp = str( Path(args.input_obs_file).with_suffix('.wp.txt') )


	## read in data

	data = np.genfromtxt(args.input_obs_file)
	meanrp = data[:, 0]
	wp = data[:, 1]
	wp_err = data[:, 2]
	ds = data[:, 3]
	ds_err = data[:, 4]
	rpmin = data[:, 5]
	rpmax = data[:, 6]


	## create scale mask

	scale_mask = np.logical_and( rpmin >= args.scale_min, rpmax <= args.scale_max )
	print(f"outputting {np.count_nonzero(scale_mask)} bins...")


	## save data

	np.savetxt(output_wp, np.c_[rpmin[scale_mask], rpmax[scale_mask],
								wp_err[scale_mask], wp[scale_mask]])

	np.savetxt(output_ds, np.c_[rpmin[scale_mask], rpmax[scale_mask],
								ds_err[scale_mask], ds[scale_mask]])