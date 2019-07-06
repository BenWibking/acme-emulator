#!/usr/bin/env python

import numpy as np
import pandas as pd
import argparse
import requests
import pycurl
import os
import io
import sys
import progressbar
import tarfile
import datetime
import shutil
import h5py as h5
from bs4 import BeautifulSoup
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath, Path

parser = argparse.ArgumentParser()
parser.add_argument('url')
parser.add_argument('redshift_dir')
parser.add_argument('download_path')
parser.add_argument('--dry-run', default=False, action='store_true')
parser.add_argument('--phases', default=False, action='store_true')
args = parser.parse_args()


class ProgressFileObject(io.FileIO):

	def __init__(self, path, progressbar, *args, **kwargs):
	
		self._total_size = os.path.getsize(path)
		self._progressbar = progressbar
		io.FileIO.__init__(self, path, *args, **kwargs)


	def read(self, size):
	
		self._progressbar.update(self.tell())
		return io.FileIO.read(self, size)
	

def walk_urls(myurl, params={}):
	
	response = requests.get(myurl, params=params)

	if response.ok:
		response_text = response.text
	else:
		return response.raise_for_status()

	soup = BeautifulSoup(response_text, 'html.parser')

	for node in soup.find_all('a'):

		link = node.get('href')

		if link[0] == '?' or link[0] == '/':
			continue	# ignore links to self or to parent directory

		if link.endswith("/"):
			newurl = myurl + link
			yield (newurl, walk_urls(newurl))

		else:
			newurl = myurl + link
			yield (newurl, None)
	
	
def combine_particle_subsamples(download_path, extracted_files, size_bytes, output_name=None):
	
	"""cat files in 'extracted_files' together into new file, then delete extracted_files."""

	output_file = download_path.with_name(output_name)
	expected_output_size = np.sum(size_bytes)
	
	## check whether output_file already exists *and* is complete
	
	if os.path.exists(output_file) and os.path.getsize(output_file) == expected_output_size:
		print(f"\tSkipping files, already combined.")
		return
	
	## combine files

	print("\tcombining extracted files...")	
	print(f"\t\toutput to: {output_file}")
	bar = progressbar.DataTransferBar(max_value=expected_output_size)
	
	with open(output_file, 'wb') as output_fp:
	
		bar.update(0)
		
		for i, filename in enumerate(extracted_files):
		
			input_file = download_path.with_name(str(filename))
			
			with open(input_file, 'rb') as input_fp:
				shutil.copyfileobj(input_fp, output_fp)
				
			bar.update(np.sum(size_bytes[:i]))
				
	bar.finish()
	
	## delete extracted_files
	
	for filename in extracted_files:
		input_file = download_path.with_name(str(filename))
		input_file.unlink()
	
	
def compute_total_size_hdf5(dsname, input_filenames, output_filename):

	total_size = 0
	for input_filename in input_filenames:
		with h5.File(input_filename, mode='r') as h5in:
			total_size += h5in[dsname].size
			
	return total_size


def get_input_dtype_hdf5(dsname, input_filenames):

	with h5.File(input_filenames[0], mode='r') as h5in:
		return h5in[dsname].dtype


def concat_hdf5(dsname, input_filenames, output_filename):

	total_size = compute_total_size_hdf5(dsname, input_filenames, output_filename)
	input_dtype = get_input_dtype_hdf5(dsname, input_filenames)
	
	with h5.File(output_filename, mode='w') as h5out:
	
		h5out.create_dataset(dsname, (total_size,),
							 dtype=input_dtype, chunks=True, compression="gzip")
							 # auto-chunksize is about 512 rows
		
		idx = 0
		for input_filename in input_filenames:
			with h5.File(input_filename, mode='r') as h5in:		# this currently hangs the process...
				data = h5in[dsname]
				h5out[dsname][idx:idx+data.size] = data
				idx += data.size
				h5out.flush()	
	
	
def combine_rockstar_halos(download_path, extracted_files, output_name=None):

	"""combine rockstar halo files into file 'output_name'."""
	
	## check if output file exists and has IS_COMPLETE attribute set to True
	
	output_filename = download_path.with_name(output_name)
	
	if os.path.exists(output_filename):
		with h5.File(output_filename, mode='r') as h5out:
			if "halos" in h5out.keys():
				dset = h5out["halos"]
				if 'IS_COMPLETE' in dset.attrs.keys() and dset.attrs['IS_COMPLETE'] == True:
					print(f"\tSkipping file, already combined.")
					return
	
	## combine files
	
	print(f"\tCombining files...")
	print(f"\t\toutput file: {output_filename}")
	concat_hdf5("halos", [download_path.with_name(f) for f in extracted_files], output_filename)
	
	## write IS_COMPLETE attribute
	
	with h5.File(output_filename, mode='a') as h5out:
		dset = h5out["halos"]
		dset.attrs['IS_COMPLETE'] = True
		h5out.flush()
		
	## delete extracted_files
	
	print(f"\tDeleting segments...")
	for filename in extracted_files:
		input_file = download_path.with_name(str(filename))
		input_file.unlink()		
	

if __name__ == '__main__':
	
	print(f"Downloading and processing simulation files. Time is {datetime.datetime.now()}\n")
	
	## create listing of all files to download by walking the directory listing

	redshift_dir_suffix = "/"
	assert args.redshift_dir.endswith(redshift_dir_suffix)
	redshift_dir = args.redshift_dir[:-len(redshift_dir_suffix)]

	data_path = Path(args.download_path)

	sim_boxes = walk_urls(args.url)
	
	for box_url, box_listing in sim_boxes:
		for subdir_url, subdir_listing in box_listing:
			for redshift_url, redshift_listing in subdir_listing:
			
				if redshift_url.endswith(args.redshift_dir):
					for data_product_url, data_product_listing in redshift_listing:
					
						## TODO: generate a list of files to download, then process list with `map`
						
						## Download header, halos.tar.gz for Rockstar subdirs;
						## download field_subsamples.tar.gz, halo_subsamples.tar.gz for FOF subdirs.
						## Raise an error if any of these files are missing.
						
						subdir_path = PurePosixPath(unquote(urlparse(subdir_url).path))
						subdir_name = subdir_path.name
						
						subdir_parent = subdir_path.parent.name
						parent_suffix = "_products"
						assert subdir_parent.endswith(parent_suffix)
						subdir_parent_name = subdir_parent[:-len(parent_suffix)]
						
						subdir_prefix = subdir_parent_name + "_"
						assert subdir_name.startswith(subdir_prefix)
						subdir_shortname = subdir_name[len(subdir_prefix):]
						
						simsuite_dir = subdir_path.parent.parent.name
						assert simsuite_dir.endswith(parent_suffix)
						simsuite_name = simsuite_dir[:-len(parent_suffix)]
						
						box_name = subdir_parent_name[len(simsuite_name):]
						if args.phases == True:
							boxname_prefix = "_00-"
						else:
							boxname_prefix = "_"
							
						assert box_name.startswith(boxname_prefix)
						box_name = box_name[len(boxname_prefix):]
						
						# make box_name a two-digit number w/ leading zeros
						box_name = box_name.zfill(2)
						
						product_dict = {'rockstar_halos': "Rockstar",
										'FoF_halos': "FOF",
										'power': "power"}
										
						assert subdir_shortname in product_dict.keys()
						subdir_product = product_dict[subdir_shortname]
						
						data_product_name = PurePosixPath(unquote(urlparse(data_product_url).path)).name
						
						FOF_files = ['field_subsamples.tar.gz', 'halo_subsamples.tar.gz']
						Rockstar_files = ['header', 'halos.tar.gz']
						
						if subdir_product == 'FOF':
							product_files = FOF_files
						elif subdir_product == 'Rockstar':
							product_files = Rockstar_files
						else:
							product_files = []
							
						if data_product_name not in product_files:
							continue
						
						download_path = data_path/simsuite_name/subdir_product/box_name/redshift_dir/data_product_name
						print(f"URL: {data_product_url}")
						print(f"filename: {download_path}")
						print("")
						
						
						## Create task for each file to do the following (depending on filename):
						
						# use this to just print inputs/outputs without downloading/moving/deleting files
						if args.dry_run:
							continue
							
							
						## i) Save file to disk.
						
						if not os.path.exists(download_path.parent):
							os.makedirs(download_path.parent)
						
						# check if file is already completely downloaded:
						c = pycurl.Curl()
						c.setopt(c.URL, data_product_url)
						c.setopt(c.NOBODY, 1) # get headers only
						c.perform()
						remote_filesize = c.getinfo(c.CONTENT_LENGTH_DOWNLOAD)
						
						download_log = download_path.with_suffix('.log')
					
						if (os.path.exists(download_path) \
							and os.path.getsize(download_path) == remote_filesize) \
							or os.path.exists(download_log):

							print("\tSkipping file, already downloaded.")
								
						else:	
							curl = pycurl.Curl()
							curl.setopt(pycurl.URL, data_product_url)
							curl.setopt(pycurl.FOLLOWLOCATION, 1)
							curl.setopt(pycurl.MAXREDIRS, 5)
							curl.setopt(pycurl.CONNECTTIMEOUT, 30)
							
							if os.path.exists(download_path):
								fp = open(download_path, 'ab')
								curl.setopt(pycurl.RESUME_FROM, os.path.getsize(download_path))
								
							else:
								fp = open(download_path, 'wb')
							
							bar = progressbar.DataTransferBar(max_value=remote_filesize)
							
							initial_size = os.path.getsize(download_path)
							
							def progress(total, existing, upload_t, upload_d):
								downloaded = existing + initial_size
								bar.update(downloaded)
															
							curl.setopt(pycurl.NOPROGRESS, 0)
							curl.setopt(pycurl.PROGRESSFUNCTION, progress)
							curl.setopt(pycurl.WRITEDATA, fp)
							
							print("\tdownloading...")
							curl.perform()
	
							curl.close()
							fp.close()
							bar.finish()
													
						
						## ii) Extract tarball, then delete tarball.

						if download_path.name.endswith('tar.gz') and not os.path.exists(download_log):
						
							print("\textracting...")
							
							bar = progressbar.DataTransferBar(max_value=os.path.getsize(download_path))
							tar = tarfile.open(fileobj=ProgressFileObject(download_path, bar))
							tar.extractall(path=download_path.parent)
							bar.finish()
							
							print("\tsaving log and deleting tarball...")
							
							logfile = open(download_log, 'w')
							logfile.write(f"# Extracted tarball '{download_path.name}' on {datetime.datetime.now()}\n")
							for file in tar.getmembers():
								logfile.write(f"{file.name}\t{file.size}\n")
								
							tar.close()
							logfile.close()
							download_path.unlink()
						
						else:
						
							print("\tSkipping file, already extracted.")
						
						
						## iii) Combine files
						
						if download_path.name.endswith('tar.gz'):
						
							download_log_df = pd.read_csv(download_log, sep='\t', skiprows=1,
															names=('files','size'))
							extracted_files = download_log_df['files']
							size_bytes = download_log_df['size']
							
							
							if download_path.name == 'field_subsamples.tar.gz':
								combine_particle_subsamples(download_path, extracted_files, size_bytes,
															output_name="field_subsamples.bin")
								
							elif download_path.name == 'halo_subsamples.tar.gz':
								combine_particle_subsamples(download_path, extracted_files, size_bytes,
															output_name="halo_subsamples.bin")
								
							elif download_path.name == 'halos.tar.gz':
								combine_rockstar_halos(download_path, extracted_files,
													   output_name="RShalos_allprops.hdf5")


						## clean up
						
						print("")