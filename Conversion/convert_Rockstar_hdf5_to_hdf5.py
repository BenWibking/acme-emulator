#!/usr/bin/env python

import numpy as np
import h5py as h5
import argparse
import config
import sys

from pathlib import Path

dsname = "halos"

halo_output_dtype = np.dtype([("N",np.int32),
							  ("x",np.float32),
							  ("y",np.float32),
							  ("z",np.float32),
							  ("vx",np.float32),
							  ("vy",np.float32),
							  ("vz",np.float32),
							  ('gid',np.int64),
							  ('mass',np.float32),
							  ('m200b_inconsistent',np.float32),
							  ('rvir',np.float32),
							  ('rs',np.float32),
							  ('vrms',np.float32)])


def convert_hdf5(input_filename, output_filename):

	input_h5_filename = input_filename
	h5_file = h5.File(input_h5_filename)
	h5_halos = h5_file['halos']

	## select only parent halos based on identification in hdf5 files
	
	halos_filtered = h5_halos[h5_halos['parent_id'] == -1]

	num_output_halos = halos_filtered.shape[0]
	print(f"num halos = {num_output_halos}", file=sys.stderr) 

	halos_output = np.empty((num_output_halos,), dtype=halo_output_dtype)

	halos_output['gid'] = halos_filtered['id']
	
	halos_output['mass'] = halos_filtered['m_SO'] # mvir_SO
	halos_output['N'] = halos_filtered['N_SO'] # particle count for mvir_SO
	
	## this leads to inconsistent halo exclusion!!
	halos_output['m200b_inconsistent'] = halos_filtered['alt_m_SO'][:,0] # m200b_SO
	
	halos_output['x'] = halos_filtered['pos'][:,0]		# Mpc/h (comoving)
	halos_output['y'] = halos_filtered['pos'][:,1]		# Mpc/h (comoving)
	halos_output['z'] = halos_filtered['pos'][:,2]		# Mpc/h (comoving)

	halos_output['vx'] = halos_filtered['vel'][:,0]		# km/s (physical)
	halos_output['vy'] = halos_filtered['vel'][:,1]		# km/s (physical)
	halos_output['vz'] = halos_filtered['vel'][:,2]		# km/s (physical)

	halos_output['rvir'] = halos_filtered['r']			# kpc/h (comoving)
	halos_output['rs'] = halos_filtered['klypin_rs']	# kpc/h (comoving)
	halos_output['vrms'] = halos_filtered['vrms']		# km/s (physical)

	del halos_filtered

	## save to hdf5
	
	with h5.File(output_filename,'w') as h5f:
		h5f.create_dataset(dsname, (num_output_halos,), dtype=halo_output_dtype, 
						   data=halos_output, chunks=True, shuffle=True, compression="gzip")
		h5f.flush()

	return num_output_halos


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='convert Rockstar .h5 format to hdf5 catalog file.')
	parser.add_argument('input_filename')
	parser.add_argument('output_filename')
	args = parser.parse_args()

	convert_hdf5(args.input_filename, args.output_filename)
