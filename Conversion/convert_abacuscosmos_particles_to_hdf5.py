#!/usr/bin/env python
import numpy as np
import h5py as h5
import argparse
import config

def get_particle_chunk(particles_mmap,begin,end,boxSize,vel_to_kms):
    particles = particles_mmap[begin:end].copy()

    # NEW FILE FORMAT DOES *NOT* USE THESE UNITS
#    particles['x'] *= boxSize
#    particles['y'] *= boxSize
#    particles['z'] *= boxSize
#    particles['vx'] *= vel_to_kms
#    particles['vy'] *= vel_to_kms
#    particles['vz'] *= vel_to_kms

    # very necessary to wrap box coordinates!
    particles['x'] = particles['x'] % boxSize
    particles['y'] = particles['y'] % boxSize
    particles['z'] = particles['z'] % boxSize

    return particles

def convert(filename, config_filename, output_filename):
    dsname = "particles"

    particle_dtype = np.dtype([('x',np.float32),('y',np.float32),('z',np.float32),('vx',np.float32),('vy',np.float32),('vz',np.float32)])

    print("converting "+filename)

    fp = open(filename,"rb")
    cf = config.AbacusConfigFile(config_filename)
    particles = np.memmap(filename, dtype=particle_dtype, mode='r')
    npart = particles.shape[0]
    print("npart: "+str(npart))

    chunk_size = 1000000
    final_array = np.array([npart,])
    chunk_flat_array = np.hstack((np.arange(0, npart, chunk_size), final_array))
    chunk_array = list(zip(chunk_flat_array[0:-1],chunk_flat_array[1:]))

    with h5.File(output_filename,'w',libver='latest') as h5f:
        dset = h5f.create_dataset(dsname, (npart,), dtype=particle_dtype, 
                           chunks=True, compression="gzip")

        for begin, end in chunk_array:
            particle_chunk = get_particle_chunk(particles,begin,end,cf.boxSize,cf.vel_to_kms)
            dset[begin:end] = particle_chunk
            h5f.flush()

    return npart

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='convert rvfloat format to hdf5 particle file.')
    parser.add_argument('particle_filename')
    parser.add_argument('header_filename')
    parser.add_argument('output_filename')
    args = parser.parse_args()

    convert(args.particle_filename, args.header_filename, args.output_filename)

