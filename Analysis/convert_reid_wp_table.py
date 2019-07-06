import numpy as np

if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser()
	
	parser.add_argument('input_wp')
	parser.add_argument('output_wp')
	
	args = parser.parse_args()
	
	
	## convert reid wp table to Emulator-Pipeline standard format
	
	wp_bins, mock_wp, mock_wp_err = np.loadtxt(args.input_wp, unpack=True)

	nbins = 20
	bin_edges = np.logspace(np.log10(0.1), np.log10(30.0), nbins+1)
	binmin = bin_edges[:-1]
	binmax = bin_edges[1:]
	binmin = binmin[2:]
	binmax = binmax[2:]
	assert np.logical_and(np.all(wp_bins > binmin), np.all(wp_bins < binmax))
	print(f"binmin.shape = {binmin.shape}")
	
	
	## save output file
	
	np.savetxt(args.output_wp, np.c_[binmin, binmax, mock_wp_err, mock_wp], delimiter='\t')
	