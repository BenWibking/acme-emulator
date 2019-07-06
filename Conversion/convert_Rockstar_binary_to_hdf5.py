#!/usr/bin/env python
import numpy as np
import h5py as h5
import argparse
import config

from pathlib import Path

dsname = "halos"

header_dtype = np.dtype([('magic', np.uint64),
                         ('snap', np.int64), ('chunk', np.int64),
                         ('scale', np.float32), ('Om', np.float32), ('Ol', np.float32), ('h0', np.float32),
                         ('bounds', np.float32, 6),
                         ('num_halos', np.int64), ('num_particles', np.int64),
                         ('box_size', np.float32), ('particle_mass', np.float32),
                         ('particle_type', np.int64),
                         ('format_revision', np.int32),
                         ('rockstar_version', 'S12'),
                         ('unused', np.uint8, 256-112)], align=True)

halo_input_dtype = np.dtype([('id',np.int64),
                             ('pos',np.float32,3),('vel',np.float32,3),
                             ('corevel',np.float32,3),('bulkvel',np.float32,3),
                             ('m',np.float32),
                             ('r',np.float32),
                             ('child_r',np.float32),
                             ('vmax_r',np.float32),
                             ('mgrav',np.float32),
                             ('vmax',np.float32),
                             ('rvmax',np.float32),
                             ('rs',np.float32),
                             ('klypin_rs',np.float32),
                             ('vrms',np.float32),
                             ('J',np.float32,3),
                             ('energy',np.float32),
                             ('spin',np.float32),
                             ('alt_m',np.float32,4),
                             ('Xoff',np.float32),
                             ('Voff',np.float32),
                             ('b_to_a',np.float32),
                             ('c_to_a',np.float32),
                             ('A',np.float32,3),
                             ('b_to_a2',np.float32),
                             ('c_to_a2',np.float32),
                             ('A2',np.float32,3),
                             ('bullock_spin',np.float32),
                             ('kin_to_pot',np.float32),
                             ('m_pe_b',np.float32),
                             ('m_pe_d',np.float32),
                             ('halfmass_radius',np.float32),
                             ('num_p',np.int64),
                             ('num_child_particles',np.int64),
                             ('p_start',np.int64),
                             ('desc',np.int64),
                             ('flags',np.int64),
                             ('n_core',np.int64),
                             ('other_floats3', np.float32, 3)], align=True)

halo_output_dtype = np.dtype([("N",np.int32,1),("x",np.float32,1),("y",np.float32,1),("z",np.float32,1),("vx",np.float32,1),("vy",np.float32,1),("vz",np.float32,1),('gid',np.int64),('mass',np.float32),('rvir',np.float32),('rs',np.float32),('vrms',np.float32)])

def convert_bin(input_filename, output_filename):
    with open(input_filename,"rb") as fp:
        bin_header = np.fromfile(fp,dtype=header_dtype,count=1)
        bin_halos = np.fromfile(fp,dtype=halo_input_dtype)
    
    boxsize = bin_header['box_size']
    
    input_h5_filename = input_filename.replace('.bin','.h5')
    h5_file = h5.File(input_h5_filename)
    h5_halos = h5_file['halos'][:]

    h5_halos.sort(order='id')
    bin_halos.sort(order='id')

    key = 'm'
    assert(not (h5_halos[key] == bin_halos[key]).all()) # the masses should be distinct

    # select only parent halos based on identification in hdf5 files
    halos_filtered = bin_halos[h5_halos['parent_id'] == -1]
    num_output_halos = halos_filtered.shape[0]
    h5_file.close()
    del h5_halos
    del bin_halos

    halos_output = np.empty((num_output_halos,),dtype=halo_output_dtype)

    halos_output['gid'] = halos_filtered['id']
#    halos_output['mass'] = halos_filtered['m'] #mvir
    halos_output['mass'] = halos_filtered['alt_m'][:,0] #m200b
    halos_output['N'] = halos_filtered['num_p']

    halos_output['x'] = (halos_filtered['pos'][:,0] + boxsize/2.0) % boxsize
    halos_output['y'] = (halos_filtered['pos'][:,1] + boxsize/2.0) % boxsize
    halos_output['z'] = (halos_filtered['pos'][:,2] + boxsize/2.0) % boxsize

    halos_output['vx'] = halos_filtered['vel'][:,0]
    halos_output['vy'] = halos_filtered['vel'][:,1]
    halos_output['vz'] = halos_filtered['vel'][:,2]

    halos_output['rvir'] = halos_filtered['r']
    halos_output['rs'] = halos_filtered['klypin_rs']
    halos_output['vrms'] = halos_filtered['vrms']

    del halos_filtered

    # save to hdf5
    with h5.File(output_filename,'w') as h5f:
        h5f.create_dataset(dsname, (num_output_halos,), dtype=halo_output_dtype, 
                           data=halos_output, chunks=True, compression="gzip")
        h5f.flush()

    return num_output_halos

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='convert Rockstar .bin format to hdf5 catalog file.')
    parser.add_argument('input_filename')
    parser.add_argument('output_filename')
    args = parser.parse_args()

    convert_bin(args.input_filename, args.output_filename)
