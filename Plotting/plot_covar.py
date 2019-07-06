import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def main(cov_filename, signal_filename, output_corr_filename, output_diag_filename, title):

	cov = np.loadtxt(cov_filename)
#    centers,signal = np.loadtxt(signal_filename,unpack=True)
	binmin,binmax,err,signal = np.loadtxt(signal_filename, unpack=True)
	centers = 0.5*(binmin + binmax)

	rmin = np.log10(binmin[0])
	rmax = np.log10(binmax[-1])

	# correlation matrix
	corr = np.zeros(cov.shape)
	for i in range(corr.shape[0]):
		for j in range(corr.shape[1]):
			corr[i,j] = cov[i,j] / np.sqrt( cov[i,i] * cov[j,j] )

	# fractional uncertainty
	fractional_error = np.sqrt(np.diag(cov)) / signal

	# plot correlation matrix
	plt.figure()
	ax = plt.gca()
	im = ax.imshow(corr,interpolation='nearest',origin='lower',
					extent=[rmin,rmax,rmin,rmax], vmin=0., vmax=1., cmap=plt.get_cmap('viridis'))
	ax.set_xlabel(r'log $r_p$ ($h^{-1}$ Mpc)')
	ax.set_ylabel(r'log $r_p$ ($h^{-1}$ Mpc)')
	ax.set_title(title)

	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.1)
	plt.colorbar(im, cax=cax)
	plt.tight_layout()
	plt.savefig(output_corr_filename)

	# plot diagonal covariances
	print(f"fractional uncertainty: {np.sqrt(np.diag(cov))/signal}")
	
	if output_diag_filename is not None:
		print(f"{output_diag_filename}")
		plt.figure()
		plt.errorbar(centers,signal,yerr=np.sqrt(np.diag(cov)),label='predicted signal')
		plt.xscale('log')
		plt.yscale('log')
		plt.ylabel('fiducial signal')
		plt.xlabel(r'$r_p$ ($h^{-1}$ Mpc)')
		plt.xlim(centers[0],centers[-1])
		plt.tight_layout()
		plt.savefig(output_diag_filename)



if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('cov_filename')
	parser.add_argument('signal_filename', help='used for radial bins')
	parser.add_argument('--output_corr', default=None, required=True)
	parser.add_argument('--output_diag', default=None)
	parser.add_argument('title')
	args = parser.parse_args()
	
	main(args.cov_filename, args.signal_filename,
	     args.output_corr, args.output_diag, args.title)
