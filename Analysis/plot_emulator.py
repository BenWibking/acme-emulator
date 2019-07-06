#!/usr/bin/env python

import argparse
import configparser
import numpy as np
import matplotlib.pyplot as plt
import re
import h5py
import compute_hod

from pathlib import Path


def tex_escape(text):

	"""
		:param text: a plain text message
		:return: the message escaped to appear correctly in LaTeX
	"""

	conv = {
		'&': r'\&',
		'%': r'\%',
		'$': r'\$',
		'#': r'\#',
		'_': r'\_',
		'{': r'\{',
		'}': r'\}',
		'~': r'\textasciitilde{}',
		'^': r'\^{}',
		'\\': r'\textbackslash{}',
		'<': r'\textless',
		'>': r'\textgreater',
	}
	
	regex = re.compile('|'.join(re.escape(str(key)) for key in \
					   sorted(conv.keys(), key = lambda item: - len(item))))
	
	return regex.sub(lambda match: conv[match.group()], text)
		
		
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	
	parser.add_argument('input_emu_filename')
	
	parser.add_argument('output_plot_cv_samples')
	parser.add_argument('output_plot_cv_covariance')

	parser.add_argument('--output_wp_accuracy', default=None)
	parser.add_argument('--output_wp_accuracy_covariance', default=None)
	
	args = parser.parse_args()
	
	
	## read emulator data
	
	f = h5py.File(args.input_emu_filename)

	gp_kernel_name = f['gp_kernel_name']
	sims_dir = f['simulations_dir']
	redshift_dir = f['redshift_dir']
	obs_filename_ext = f['filename_extension_inputs']
	param_files = [b.decode('utf-8') for b in f['param_files_inputs']]
	param_names = [b.decode('utf-8') for b in f['param_names_inputs']]
	
	binmin = f['rbins_min'][:]
	binmax = f['rbins_max'][:]
	binmed = 0.5*(binmin+binmax)
	
	x = f['raw_training_inputs']
	y = f['raw_training_outputs']
	yerr = f['err_raw_training_outputs']
	
	X = f['normalized_training_inputs'][:]
#	err_vectors = f['looe_err_vectors'][:]
	err_vectors = f['simfold_err_vectors'][:]
	
	y_mean = f['mean_training_outputs'][:]
	y_sigma = f['stdev_training_outputs'][:]
	
	
	## plot cross-validation error as a function of parameters

#	j_bin_min = 0
#	j_bin_max = 10		# average errors for bins 0-9
#	this_err_vector = np.mean(y_sigma[j_bin_min:j_bin_max, np.newaxis] * err_vectors[j_bin_min:j_bin_max, :], axis=0)
#	
#	corner_fig = corner.corner(x,
#							labels=[tex_escape(s) for s in param_names],
#							plot_contours = True,
#							fill_contours = False,
#							no_fill_contours = True,
#							contour_kwargs = {'colors': ['black','green','blue','red']},
#							plot_density = False,
#							plot_datapoints = False,
#							show_titles = True,
#							levels = [0.01, 0.05, 0.1, 0.2],
#							weights = abs(this_err_vector))
#							
#	corner_fig.savefig(args.output_plot_corner)
	
	
	## plot prediction errors for random sub-set

	plt.figure()

	dimensionful_err_vectors = np.empty(err_vectors.shape)
	frac_err_vectors = np.empty(err_vectors.shape)

	for i in range(err_vectors.shape[1]):
#		dimensionful_err_vectors[:,i] = y_sigma*err_vectors[:,i]	# only for LOOE
		dimensionful_err_vectors[:,i] = err_vectors[:,i]			# for sim-fold errors
		frac_err_vectors[:,i] = dimensionful_err_vectors[:,i] / y_mean

	nplot = frac_err_vectors.shape[1]
	for i in range(nplot):
		this_err_vector = frac_err_vectors[:,i]
		plt.plot(binmed, this_err_vector, alpha=0.2, color='grey')

	max_err_vector = np.max(frac_err_vectors, axis=1)
	min_err_vector = np.min(frac_err_vectors, axis=1)
	pct97_err_vector = np.percentile(frac_err_vectors, 97.0, axis=1)
	pct85_err_vector = np.percentile(frac_err_vectors, 85.0, axis=1)
	pct15_err_vector = np.percentile(frac_err_vectors, 15.0, axis=1)
	pct03_err_vector = np.percentile(frac_err_vectors, 3.0, axis=1)
	avg_err_vector = np.mean(frac_err_vectors, axis=1)
	std_err_vector = np.std(frac_err_vectors, axis=1)

	plt.fill_between(binmed, avg_err_vector-std_err_vector, avg_err_vector+std_err_vector,
					 facecolor='none', hatch='X', edgecolor='black', linewidth=0.0,
					 zorder=20, label='mean/std. dev.')
					 

	## plot LOWZ error bars

#	cov = np.loadtxt('../../lowz_mocks/Ben_gg_covariance_z0p3.txt')	# wp covariance
#	subvolume = (1100.)**3 / 25.0
#	vol_survey = (913.26)**3
#	cov *= (20*25) * (subvolume / vol_survey)
#	wpmin, wpmax, _, wp_fiducial = np.loadtxt('./Params/LOWZ_phases_03/NHOD_lowz_fiducial.00.0.seed_42.template_param.average_wp.txt', unpack=True)
#	cov_err = np.sqrt( np.diag(cov) ) / wp_fiducial

	cov = np.loadtxt('./Params/LOWZ_emulator_03/lowz_cov_deltasigma_new.txt')
	dsmin, dsmax, _, ds_fiducial = np.loadtxt('./Params/LOWZ_phases_03/NHOD_lowz_fiducial.00.0.seed_42.template_param.average_DeltaSigma.txt', unpack=True)
	cov_err = np.sqrt( np.diag(cov) ) / ds_fiducial

	for i in range(len(binmed)):
		print(f"{i} {binmin[i]} {binmax[i]} {std_err_vector[i]/cov_err[i]}")

	plt.fill_between(binmed, -cov_err, cov_err,
					 color='orange', alpha=0.5, label='LOWZ error bars', zorder=100)


	## plot emulator errors

	plt.plot(binmed, avg_err_vector, '-', color='black', zorder=21)
	plt.plot(binmed, min_err_vector, '--', color='red', zorder=21, label='min/max error')
	plt.plot(binmed, max_err_vector, '--', color='red', zorder=21)
	plt.plot(binmed, pct15_err_vector, '--', color='black', zorder=21, label='15th/85th percentile')
	plt.plot(binmed, pct85_err_vector, '--', color='black', zorder=21)
	plt.plot(binmed, pct03_err_vector, '--', color='blue', zorder=21, label='3rd/97th percentile')
	plt.plot(binmed, pct97_err_vector, '--', color='blue', zorder=21)

	plt.xscale('log')
	plt.ylabel(r'leave-one-simulation-out fractional error')
	plt.xlabel(r'$r_p$ [$h^{-1}$ Mpc]')
	plt.legend(loc='best')
	plt.xlim((binmin[0], binmax[-1]))
#	plt.ylim(-0.15, 0.15)
	plt.tight_layout()
	plt.savefig(args.output_plot_cv_samples)
	plt.close()
	
	
	## plot correlation matrix of prediction errors
	
	cov = np.cov(dimensionful_err_vectors)
	print(f'covariance matrix: {cov.shape[0]} x {cov.shape[1]}')
	corr = np.empty(cov.shape)
	
	for i in range(corr.shape[0]):
		for j in range(corr.shape[1]):
			corr[i,j] = cov[i,j] / np.sqrt(cov[i,i]*cov[j,j])
	
	from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
	plt.figure()
	ax = plt.gca()

	im = ax.imshow(np.swapaxes(corr, 0, 1), vmin=-1.0, vmax=1.0, cmap='Spectral', aspect='equal',
					extent=[np.log10(binmin[0]),np.log10(binmax[-1]),
							np.log10(binmin[0]),np.log10(binmax[-1])])

	ax.set_xticks(np.arange(-1.0, 1.5, 1.0))
	ax.set_yticks(np.arange(-1.0, 1.5, 1.0))

	divider = make_axes_locatable(ax)
	width = axes_size.AxesY(ax, aspect=1/20)
	pad = axes_size.Fraction(1.0, width)
	cax = divider.append_axes('right', size=width, pad=pad)
	plt.colorbar(im, cax=cax, ticks=[-1,0,1])

	ax.set_xlabel(r'$\log_{10} r$ [$h^{-1}$ Mpc]')
	ax.set_ylabel(r'$\log_{10} r$ [$h^{-1}$ Mpc]')

	plt.tight_layout()
	plt.savefig(args.output_plot_cv_covariance)


