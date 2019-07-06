#!/usr/bin/env python

import numpy as np
from pathlib import Path
import re
import sys
import argparse
import tarfile

parser = argparse.ArgumentParser()
parser.add_argument('--rootdir', required=True)
parser.add_argument('--redshift-dir', required=True)
parser.add_argument('--dry-run', default=False, action='store_true')
args = parser.parse_args()

dry_run = args.dry_run	# use this to just print inputs/outputs without moving/deleting files

root = Path(args.rootdir)

FOF_tarballs = ['field_subsamples.tar.gz', 'halo_subsamples.tar.gz']
Rockstar_tarballs = ['halos.tar.gz']	

for halotype in ['FOF', 'Rockstar']:
	thisroot = root / halotype
	subdirs = [x for x in thisroot.iterdir() if x.is_dir()]

	for subdir in subdirs:
		simdir = subdir / args.redshift_dir
		print(simdir)
		
		tarballs = []
		if halotype == 'FOF':
			tarballs = FOF_tarballs
		elif halotype=='Rockstar':
			tarballs = Rockstar_tarballs

		# extract and delete tarballs one-by-one
		for tarball_filename in tarballs:
			fullpath = Path(simdir / tarball_filename)
			tar = tarfile.open(str(fullpath), mode='r')

			if args.dry_run == False:
				print("extracting...")
				tar.extractall(path=str(Path(simdir)))
				tar.close()
				print("done. deleting.")
				fullpath.unlink()

		print("")
