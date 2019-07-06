#!/usr/bin/env python
import numpy as np
import h5py as h5
import argparse
import config

def convert(filename, config_filename, output_filename):
    dsname = "particles"

    particle_dtype = np.dtype([('x',np.float32),('y',np.float32),('z',np.float32),('vx',np.float32),('vy',np.float32),('vz',np.float32)])

    print("converting "+filename)

    fp = open(filename,"rb")
    cf = config.AbacusConfigFile(config_filename)
    particles = np.fromfile(fp,dtype=particle_dtype)
    npart = particles.shape[0]

    print("npart: "+str(npart))

    particles['x'] *= cf.boxSize
    particles['y'] *= cf.boxSize
    particles['z'] *= cf.boxSize

    particles['x'] += cf.boxSize/2.
    particles['y'] += cf.boxSize/2.
    particles['z'] += cf.boxSize/2.

    particles['vx'] *= cf.vel_to_kms
    particles['vy'] *= cf.vel_to_kms
    particles['vz'] *= cf.vel_to_kms

    with h5.File(output_filename,'w') as h5f:
        h5f.create_dataset(dsname, (npart,), dtype=particle_dtype, 
                           data=particles, chunks=True, compression="gzip")
        h5f.flush()

    return npart

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='convert rvfloat format to hdf5 particle file.')
    parser.add_argument('particle_filename')
    parser.add_argument('header_filename')
    parser.add_argument('output_filename')
    args = parser.parse_args()

    convert(args.particle_filename, args.header_filename, args.output_filename)

