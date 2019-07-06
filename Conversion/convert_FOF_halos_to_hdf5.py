#!/usr/bin/env python
import numpy as np
import h5py as h5
import argparse
import config

from pathlib import Path
from parse import parse

dsname = "halos"

halo_input_dtype = halo_type = np.dtype([("gid",np.int32),("N",np.int32),("x",np.float32,3),("v",np.float32,3),("sigma_v",np.float32,3),("r25",np.float32),("r50",np.float32),("r75",np.float32),("r90",np.float32),("vcirc_max",np.float32),("rvcirc_max",np.float32),("pf_start",np.int64),("pf_np",np.int32)])

halo_output_dtype = np.dtype([("N",np.int32,1),("x",np.float32,1),("y",np.float32,1),("z",np.float32,1),("vx",np.float32,1),("vy",np.float32,1),("vz",np.float32,1),("sigma_vx",np.float32,1),("sigma_vy",np.float32,1),("sigma_vz",np.float32,1),('gid',np.int64),('mass',np.float32),('vmax',np.float32)])

def convert(input_filename, header_filename, output_filename):
    # determine nslab based on input_filename
    nslab = int(parse("halos_{}",Path(input_filename).name)[0])

    cf = config.AbacusConfigFile(header_filename)

    fp = open(input_filename,"rb")
    num_input_halos = np.fromfile(fp,dtype=np.int32,count=1)
    halos = np.fromfile(fp,dtype=halo_input_dtype)

    # remove FOF halos with fewer than 20 particles
    halos_filtered = halos[halos['N']>=20] # index arrays return copies, not views, so this is ok
    num_output_halos = halos_filtered.shape[0]
    del halos # from here on, program will use less than the memory used at this point

    halos_output = np.empty((num_output_halos,),dtype=halo_output_dtype)

    # add 64-bit gid:
    # first 4 bytes is the integer slab number 'nslab'
    # last 4 bytes is the 'gid' field
    id_array = np.ndarray(shape=(num_output_halos,2),dtype='i4')
    id_array[:,0] = nslab
    id_array[:,1] = halos_filtered['slab_gid']
    gid_array = np.ndarray(shape=num_output_halos,dtype='i8',buffer=id_array)

    halos_output['gid'] = gid_array
    halos_output['mass'] = halos_filtered['N'] * cf.particleMass
    halos_output['vmax'] = halos_filted['vcirc_max']
    halos_output['N'] = halos_filtered['N']

    halos_output['x'] = halos_filtered['x'] * cf.boxSize
    halos_output['y'] = halos_filtered['y'] * cf.boxSize
    halos_output['z'] = halos_filtered['z'] * cf.boxSize

    halos_output['x'] += cf.boxSize/2.
    halos_output['y'] += cf.boxSize/2.
    halos_output['z'] += cf.boxSize/2.

    halos_output['vx'] = halos_filtered['vx']*cf.vel_to_kms
    halos_output['vy'] = halos_filtered['vy']*cf.vel_to_kms
    halos_output['vz'] = halos_filtered['vz']*cf.vel_to_kms

    halos_output['sigma_vx'] = halos_filtered['sigma_vx']*cf.vel_to_kms
    halos_output['sigma_vy'] = halos_filtered['sigma_vy']*cf.vel_to_kms
    halos_output['sigma_vz'] = halos_filtered['sigma_vz']*cf.vel_to_kms

    del halos_filtered

    # save to hdf5
    with h5.File(output_filename,'w') as h5f:
        h5f.create_dataset(dsname, (num_output_halos,), dtype=halo_output_dtype, 
                           data=halos_output, chunks=True, compression="gzip")
        h5f.flush()

    return num_output_halos

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='convert Abacus FOF format to hdf5 catalog file.')
    parser.add_argument('input_filename')
    parser.add_argument('header_filename')
    parser.add_argument('output_filename')
    args = parser.parse_args()

    convert(args.input_filename, args.header_filename, args.output_filename)
