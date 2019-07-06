#!/usr/bin/env python
import argparse
import numpy as np
import h5py as h5

def select_halos(halo_file, output_file, Np_resolved=100):
    with h5.File(halo_file, mode='r') as catalog:
        halos = catalog['halos']
        N = halos['N']
        resolved_halos = halos[N >= Np_resolved]

        # write resolved_halos to output_file
        with h5.File(output_file, mode='w') as output:
            output.create_dataset('halos', data=resolved_halos)
        # closed and written automatically


parser = argparse.ArgumentParser()
parser.add_argument('halo_path')
parser.add_argument('output_path')

args = parser.parse_args()

select_halos(args.halo_path, args.output_path)

