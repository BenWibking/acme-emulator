#!/usr/bin/env python

import argparse
import configparser
import numpy as np
import scipy.special
import h5py
from pathlib import Path
from math import exp, sqrt
from numba import jit

import compute_hod


def get_emulate_fun(input_emu_filename, load_scalar=False):

	"""return a closure that emulates as a function of input parameters."""

	## read emulator data
	
	f = h5py.File(input_emu_filename, mode='r')

	emu_filename = f['this_filename'][()]
	assert str(Path(input_emu_filename).name) == str(Path(emu_filename).name)
	
	sims_dir = f['simulations_dir'][()]
	redshift_dir = f['redshift_dir'][()]
	obs_filename_ext = f['filename_extension_inputs'][()]
	
	param_names = [b.decode('utf-8') for b in f['param_names_inputs']]
	
	gp_kernel_name = f['gp_kernel_name']
	gp_kernel_hyperparameters = f['gp_kernel_hyperparameters']
	gp_kernel_sourcecode = f['gp_kernel_sourcecode']
	
	if load_scalar == True:
		wp_binmin = f['rbins_min']
		wp_binmax = f['rbins_max']
	else:
		wp_binmin = f['rbins_min'][:]				# r_p bin edges
		wp_binmax = f['rbins_max'][:]
	
	y_mean = f['mean_training_outputs'][:]		# mean of training data
	y_sigma = f['stdev_training_outputs'][:]	# standard deviation of training data
	
	x = f['raw_training_inputs'][:]		 		# parameters for training data
	X = f['normalized_training_inputs'][:] 		# normalized parameters for training data
	
	coefs = f['best_fit_coefs']					# GP coefficients [ shape: (y.shape[0], X.shape[0]) ]

	ndata = x.shape[0]
	nparams = x.shape[1]
	nbins = y_mean.shape[0]


	## define mappings from input data to emulator inputs, from emulator output to 'real' outputs

	xmin_params = np.empty(x.shape[1])
	xmax_params = np.empty(x.shape[1])
	
	for j in range(x.shape[1]):
		xmin_params[j] = x[:,j].min()
		xmax_params[j] = x[:,j].max()
		
	range_params = xmax_params - xmin_params
		
	def normalize_parameters(parameters):
	
		"""Convert HOD+cosmo parameters to emulator inputs in the range [0, 1].
			Should be identical to 'compute_labels' in fit_data.py.
			Applying this function to x should return X."""
		
		return (parameters - xmin_params) / range_params
		
	def denormalize_outputs(emu_output):
	
		"""Convert emulator output into ratio w.r.t analytic prediction."""
		
		return y_sigma*emu_output + y_mean
		
	assert np.allclose( normalize_parameters(x), X )	# test normalize_parameters()	


	## hyperparameters
	
	hyperparams = gp_kernel_hyperparameters[:]
	lambdas = hyperparams[:, 0]
	
	if hyperparams.shape[1] > 1:
		scales = hyperparams[:, 1:]
	else:
		scales = np.zeros(hyperparams.shape[0])
	
	
	## dynamically initialize kernel function (**don't run code from untrusted files!!**)
	
	function_source = gp_kernel_sourcecode[()]

	code_obj = compile(function_source, emu_filename, 'exec')
	code_locals = dict()
	exec(code_obj, globals(), code_locals) # def function into code_locals
	function_name = list(code_locals)[0]
	
	print(f"nparams = {nparams}")
	print(f"nbins = {nbins}")
	print(f"ndata = {ndata}")
	print(f"param names: {param_names}")
	print(f"Using kernel function: {function_name}")
	kernel_fun = code_locals[function_name]


	## compute model prediction (for each bin j)

	@jit
	def emulate_fun(input_parameters):

		X_star = normalize_parameters(input_parameters)
		emu_output = np.zeros(nbins)
	
		for j in range(nbins):
			for i in range(coefs.shape[1]):
				emu_output[j] += kernel_fun(X[i,:], X_star, scales[j, :]) * coefs[j,i]

		return denormalize_outputs(emu_output)

	return emulate_fun, wp_binmin, wp_binmax



def compute_analytic_wp(omega_m, omega_b, H0, ns, w0, sigma_8, redshift,
						ngal, siglogM, M1_over_Mmin, M0_over_M1, alpha, f_cen,
						wp_binmin=None, wp_binmax=None):

	"""compute analytic wp for a given cosmology + HOD."""

	## convert parameters

	omega_c = omega_m - omega_b
	h = H0/100.
	ombh2 = omega_b*h**2
	omch2 = omega_c*h**2


	## compute (linear) power spectrum
	
	k, P = compute_hod.eisenstein_hu_pk(ombh2=ombh2, omch2=omch2, H0=H0, ns=ns, w0=w0,
										sigma8=sigma_8, redshift=redshift)

	## convenience functions for mass function fitting formulae

	dndm = lambda M: compute_hod.dndm_tinker(M, z=redshift, k=k, P=P, Omega_M=omega_m)
	bias = lambda M: compute_hod.compute_linear_bias(M, k, P, omega_m=omega_m)
	
	dndm_vec = np.vectorize(dndm)
	bias_vec = np.vectorize(bias)

	mass_tab = np.logspace(10., 16., 512)
	massfun_tab = dndm_vec(mass_tab)
	bias_tab = bias_vec(mass_tab)


	## compute logMmin from integral over Tinker mass function

	logMmin, logM0, logM1 = compute_hod.compute_HOD_parameters(ngal=ngal,
								siglogM=siglogM,
								M1_over_Mmin=M1_over_Mmin,
								M0_over_M1=M0_over_M1,
								alpha=alpha,
								f_cen=f_cen,
								mass_tabulated=mass_tab,
								massfun_tabulated=massfun_tab)

	## compute 1-halo xi_gg

	M0 = 10.**(logM0)
	M1 = 10.**(logM1)

	def Ncen_of_M(M):
		return 0.5 * (1.0 + scipy.special.erf((np.log10(M) - logMmin) / siglogM))
		
	def Nsat_of_M(M):
		Nsat = 0.
		if M > M0:
				#Nsat = Ncen_of_M(M) * ( ((M - M0)/M1)**alpha )
				Nsat = ( ((M - M0)/M1)**alpha )
		return Nsat

	def global_N_of_M(M):
		Ncen = Ncen_of_M(M)
		Nsat = Nsat_of_M(M)
		return Ncen * (1.0 + Nsat)


	cM_interpolated = lambda M: compute_hod.cM_Correa2015(M, z=redshift)

	xigg_rbins = np.logspace(np.log10(0.1), np.log10(110.0), 80)

	r, xi_gg1, xi_gg1cs, xi_gg1ss = compute_hod.compute_xi_1halo(xigg_rbins, global_N_of_M,
					 Ncen_of_M, Nsat_of_M,
					 cM=cM_interpolated, dndm=massfun_tab, mass_tab=mass_tab,
					 Omega_M=omega_m, redshift=redshift)

	r2, xi_gg2, xi_mm = compute_hod.compute_xi_2halo(xigg_rbins, global_N_of_M,
						redshift=redshift, k=k, Pk=P,
						dndm=massfun_tab, mass_tab=mass_tab,
						Omega_M=omega_m, biasfun=bias_tab)

	xigg_analytic = xi_gg1 + xi_gg2
	

	## compute wp

	rp_binmin, rp_binmax, analytic_wp = compute_hod.wp(xigg_rbins, xigg_rbins, xigg_analytic,
													   rp_binmin=wp_binmin, rp_binmax=wp_binmax)
	return analytic_wp


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	
	parser.add_argument('input_emu_filename')
	parser.add_argument('input_parameters')
	parser.add_argument('output_wp_file')
	
	args = parser.parse_args()
	
	
	## read input parameters
	
	import configparser
	config = configparser.ConfigParser()
	config.read(args.input_parameters)
	params = config['params']
	
	ngal = float(params['ngal'])
	M0_over_M1 = float(params['M0_over_M1'])
	M1_over_Mmin = float(params['M1_over_Mmin'])
	siglogM = float(params['siglogM'])
	alpha 	= float(params['alpha'])
	f_cen	= float(params['f_cen']) # NOTE: currently unused in emulator

	omega_m = float(params['omega_m'])
	sigma_8 = float(params['sigma_8'])
	H0 = float(params['H0'])
	ns = float(params['ns'])
	omegab = float(params['omega_b'])
	w0 = float(params['w0'])
	redshift = float(params['redshift'])


	## convert parameters

	omegac = omega_m - omegab
	h = H0/100.
	ombh2 = omegab*h**2
	omch2 = omegac*h**2


	## compute emulated ratio

	emulate_fun, wp_binmin, wp_binmax = get_emulate_fun(args.input_emu_filename)

	input_parameters = np.array([ngal, siglogM, M0_over_M1, M1_over_Mmin, alpha,
								sigma_8, H0, ombh2, omch2, w0, ns])
	predicted_wp_ratio = emulate_fun(input_parameters)

	print(f"predicted ratio = {predicted_wp_ratio}")
	

	## compute analytic wp
	
	analytic_wp = compute_analytic_wp(omega_m, omegab, H0, ns, w0, sigma_8,
										redshift,
										ngal, siglogM,
										M1_over_Mmin, M0_over_M1, alpha, f_cen,
										wp_binmin=wp_binmin,
										wp_binmax=wp_binmax)
										
	predicted_wp = predicted_wp_ratio * analytic_wp

	print(f"predicted wp = {predicted_wp}")
	print(f"rp = {0.5*(wp_binmin+wp_binmax)}")

	
	## output prediction to file
	
	np.savetxt(args.output_wp_file, np.c_[wp_binmin, wp_binmax, np.zeros_like(predicted_wp), predicted_wp])
