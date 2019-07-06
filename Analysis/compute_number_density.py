#!/usr/bin/env python
import argparse
import numpy as np
import h5py as h5
import sys

import config

def compute_number_density(header_file, catalog_file, output_file, centrals_only=False):
    cf = config.AbacusConfigFile(header_file)
    boxsize = cf.boxSize

    with h5.File(catalog_file, mode='r') as mock:
        vol = boxsize**3
        if centrals_only == True:
            is_sat = mock['particles']['is_sat']
            galaxies = mock['particles'][is_sat == 0]
        else:
            galaxies = mock['particles']
        ngal = np.sum(galaxies['weight']) / vol

    print('number density = {}'.format(ngal), file=sys.stderr)

    ## save mass function
    np.savetxt(output_file, np.array([ngal]))


parser = argparse.ArgumentParser()
parser.add_argument('header_file')
parser.add_argument('catalog_file')
parser.add_argument('output_file')
parser.add_argument('--centrals_only',default=False,action='store_true')

args = parser.parse_args()

compute_number_density(args.header_file, args.catalog_file, args.output_file, centrals_only=args.centrals_only)

