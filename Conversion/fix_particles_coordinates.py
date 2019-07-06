#!/usr/bin/env python
import numpy as np
import h5py as h5
import argparse
import config
from pathlib import Path
import progressbar

def get_particle_chunk(particles_mmap, begin, end, boxSize):
    particles = particles_mmap[begin:end].copy()
    particles['x'] = (particles['x'] - boxSize/2.) % boxSize
    particles['y'] = (particles['y'] - boxSize/2.) % boxSize
    particles['z'] = (particles['z'] - boxSize/2.) % boxSize
    return particles

def fix_particles_hdf5(config_filename, input_filename, output_filename):
    print(config_filename)
    cf = config.AbacusConfigFile(config_filename)
    boxsize = cf.boxSize

    dsname = "particles"
    input_file = h5.File(input_filename)
    particles = input_file[dsname]
    particle_dtype = particles.dtype
    npart = len(particles)
    print("npart: "+str(npart))
    print("converting "+input_filename)

    chunk_size = 1000000
    final_array = np.array([npart,])
    chunk_flat_array = np.hstack((np.arange(0, npart, chunk_size), final_array))
    chunk_array = list(zip(chunk_flat_array[0:-1],chunk_flat_array[1:]))

    with h5.File(output_filename,'w',libver='latest') as h5f:
        dset = h5f.create_dataset(dsname, (npart,), dtype=particle_dtype, 
                           chunks=True, compression="gzip")

        bar = progressbar.ProgressBar(max_value=len(chunk_array))
        for i, (begin, end) in enumerate(chunk_array):
            particle_chunk = get_particle_chunk(particles, begin, end, boxsize)
            dset[begin:end] = particle_chunk
            h5f.flush()
            bar.update(i)

    return npart


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fix input particles file coordinates. save as new file.')
    parser.add_argument('config_filename')
    parser.add_argument('input_filename')
    parser.add_argument('output_filename')
    args = parser.parse_args()

    fix_particles_hdf5(args.config_filename, args.input_filename, args.output_filename)
