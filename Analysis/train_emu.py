#!/usr/bin/env python

import numpy as np
import numpy.linalg
import inspect
import argparse
import configparser
import config
import time
import nlopt
import h5py
import pandas as pd
from numba import jit
from pathlib import Path

from math import exp, sqrt, log


@jit
def polynomial_kernel(X_i,X_j,h,p=None):
	dotproduct = (X_i @ np.diag(h[2:]**-2) @ X_j.T)
	return h[0] * (h[1] + dotproduct)**p


@jit
def poly2_kernel(X_i,X_j,h):
	return polynomial_kernel(X_i,X_j,h,p=2)


@jit
def poly3_kernel(X_i,X_j,h):
	return polynomial_kernel(X_i,X_j,h,p=3)


@jit
def poly4_kernel(X_i,X_j,h):
	return polynomial_kernel(X_i,X_j,h,p=4)


@jit
def poly5_kernel(X_i,X_j,h):
	return polynomial_kernel(X_i,X_j,h,p=5)


@jit
def poly6_kernel(X_i,X_j,h):
	return polynomial_kernel(X_i,X_j,h,p=6)


@jit
def matern32_kernel(X_i,X_j,h):

	"""compute Matern-3/2 kernel."""
	
	w = (X_i - X_j)
	r = sqrt(w @ (np.diag(h[1:]**-2) @ w))
	
	return h[0]*(1.0 + sqrt(3.0)*r)*exp(-sqrt(3.0)*r)


@jit
def matern52_kernel(X_i,X_j,h):

	"""compute Matern-5/2 kernel."""
	
	w = (X_i - X_j)
	r = (w @ (np.diag(h[1:]**-2) @ w))
	r = sqrt(rsq)
	
	return h[0]*(1.0 + sqrt(5.0)*r + (5./3.)*rsq)*exp(-sqrt(5.0)*r)


@jit
def poly2_matern32_kernel(X_i,X_j,h):

	scales = h[3:]
	mid = len(scales) // 2
	s1 = scales[:mid]
	s2 = scales[mid:]
	
	w = (X_i - X_j)	
	H1 = np.diag(s1**-2)
	H2 = np.diag(s2**-2)
	r = sqrt(w @ (H1 @ w))
	dotproduct = (X_i @ ( H2 @ X_j.T ))
	
	return h[0]*(1.0 + sqrt(3.0)*r)*exp(-sqrt(3.0)*r) + h[1]*(1.0 + dotproduct)**2 + h[2]


@jit
def sqrexp_matern32_kernel(X_i,X_j,h):

	"""compute Squared Exponential + Matern-3/2 kernel."""
	scales = h[3:]
	mid = len(scales) // 2
	s1 = scales[:mid]
	s2 = scales[mid:]
	
	w = (X_i - X_j)
	r1 = sqrt(w @ (np.diag(s1**-2) @ w))
	r2 = sqrt(w @ (np.diag(s2**-2) @ w))
	
	return h[0]*(1.0 + sqrt(3.0)*r1)*exp(-sqrt(3.0)*r1) + h[1]*exp(-0.5*r2*r2) + h[2]


@jit
def sqrexp_matern52_kernel(X_i,X_j,h):

	"""compute Squared Exponential + Matern-5/2 kernel."""
	scales = h[3:]
	mid = len(scales) // 2
	s1 = scales[:mid]
	s2 = scales[mid:]
	
	w = (X_i - X_j)
	r1 = sqrt(w @ (np.diag(s1**-2) @ w))
	r2 = sqrt(w @ (np.diag(s2**-2) @ w))
	
	return h[0]*(1.0 + sqrt(5.0)*r1 + (5./3.)*r1*r1)*exp(-sqrt(5.0)*r1) + \
			h[1]*exp(-0.5*r2*r2) + h[2]


@jit
def sqrexp_kernel(X_i,X_j,h):

	"""compute Squared Exponential kernel."""
	s = h[2:]	
	w = (X_i - X_j)
	rsq = w @ (np.diag(s**-2) @ w)
	
	return h[0]*exp(-0.5*rsq) + h[1]


@jit
def spline_kernel(X_i,X_j,h):

	"""compute cubic spline kernel from tensor product of scalar cubic splines.
	Parameters should be normalized to [0,1)."""
	
	product = 1.0
	for k in range(X_i.shape[0]):
		x = X_i[k]
		y = X_j[k]
		z = min(x,y)
		product *= 1.0 + x*y + 0.5*abs(x-y)*z**2 + (1./3.)*z**3

	return product


@jit
def polyharmonic_kernel(X_i,X_j,h):

	"""compute polyharmonic (thin-plate spline) kernel."""
	
	w = (X_i - X_j)
	r = sqrt(w @ w.T)
	D = X_i.shape[0]
	k = 0.
	if D % 2 == 0: # D is even
		k = r * log(r**r)
	else: # D is odd
		k = r**3
		
	return k + (1.0 + X_i @ X_j.T)**2


@jit
def compute_kernel(X,h,generic_kernel):

	K = np.empty((X.shape[0],X.shape[0])) # kernel matrix
	for i in range(K.shape[0]):
		for j in range(K.shape[1]):
			K[i,j] = generic_kernel(X[i,:],X[j,:],h)
			
	return K


@jit
def LOOE_vector(c,Ginv):

	"""return the leave-one-out error vector."""
	
	this_LOOE_vector = np.zeros(c.shape[0])
	for i in range(c.shape[0]):
		this_LOOE_vector[i] += (c[i,0] / Ginv[i,i])
	return this_LOOE_vector


@jit
def fit_data_with_kernel(Y, K, Yerr, Lambda):

	"""fit data with a least-squares model with kernel K, observations Y.
	If noise of observations is Gaussian, \Lambda = 2*var(x)."""
	
	for i in range(K.shape[0]):
		K[i,i] += Lambda * ( Yerr[i]**2 )
	
	eigval, Q = np.linalg.eigh(K)	# this is faster than computing np.inv(K)

	D = np.zeros(Q.shape)
	for i in range(D.shape[0]):
		D[i,i] = 1.0/eigval[i]

	Ginv = Q @ D @ Q.T
	c = Ginv @ Y
	
	err_vector = LOOE_vector(c,Ginv)
	N = c.shape[0]
	err_norm = np.sum(err_vector**2)
	rms_err_norm = np.sqrt(err_norm/N)
	
	## Rasmussen & Williams, ch. 5, eq. 5.10
	logp = 0.
	for i in range(Yerr.shape[0]):
		sqerr = Lambda * ( Yerr[i]**2 )
		logp_i = -0.5*np.log(sqerr) - (err_vector[i]**2)/(2.0*sqerr) - 0.5*np.log(2.0*np.pi)
		logp += logp_i

	return c, rms_err_norm, np.asscalar(logp), err_vector
	

def fit_hyperparameters(Y, X, Yerr, kernel='poly'):

	"""fit the hyperparameters of the kernel function."""

	N=Y.shape[0]
	nparams=X.shape[1]

	lambda_guess = 0.01
	lambda_max = 1.0
	lambda_min = 1.0e-4


	if kernel=='poly2' or kernel=='poly3' or kernel=='poly4' \
		or kernel=='poly5' or kernel=='poly6':
	
		scale_guess = np.ones(nparams)
		scale_max = 1e2 * scale_guess
		scale_min = 1e-2 * scale_guess
		kernel_guess = np.array([np.var(Y), np.var(Y)])
		kernel_max = 1e2 * kernel_guess
		kernel_min = 1e-2 * kernel_guess
		x0 = np.concatenate([[lambda_guess], kernel_guess, scale_guess])
		hyper_max = np.concatenate([[lambda_max], kernel_max, scale_max])
		hyper_min = np.concatenate([[lambda_min], kernel_min, scale_min])
		if kernel=='poly2':
			generic_kernel = poly2_kernel
		elif kernel=='poly3':
			generic_kernel = poly3_kernel
		elif kernel=='poly4':
			generic_kernel = poly4_kernel
		elif kernel=='poly5':
			generic_kernel = poly5_kernel
		elif kernel=='poly6':
			generic_kernel = poly6_kernel
			
	elif kernel=='spline':
	
		x0 = np.concatenate([[lambda_guess]])
		hyper_max = np.concatenate([[lambda_max]])
		hyper_min = np.concatenate([[lambda_min]])
		generic_kernel = spline_kernel
		
	elif kernel=='polyharmonic':
	
		x0 = np.concatenate([[lambda_guess]])
		hyper_max = np.concatenate([[lambda_max]])
		hyper_min = np.concatenate([[lambda_min]])
		generic_kernel = polyharmonic_kernel
		
	elif kernel=='matern32':
	
		scale_guess = np.array([(np.max(X[:,i]) - np.min(X[:,i])) for i in range(nparams)])
		scale_max = 1e5  * scale_guess
		scale_min = 1e-5 * scale_guess
		kernel_guess = np.array([np.var(Y) - lambda_guess])
		kernel_max = 1.0
		kernel_min = np.zeros_like(kernel_guess)
		x0 = np.concatenate([[lambda_guess], kernel_guess, scale_guess])
		hyper_max = np.concatenate([[lambda_max], kernel_max, scale_max])
		hyper_min = np.concatenate([[lambda_min], kernel_min, scale_min])
		generic_kernel = matern32_kernel
		
	elif kernel=='matern52':
	
		scale_guess = np.array([(np.max(X[:,i]) - np.min(X[:,i])) for i in range(nparams)])
		scale_max = 1e5  * scale_guess
		scale_min = 1e-5 * scale_guess
		kernel_guess = np.array([np.var(Y)])
		kernel_max = 1e5 * kernel_guess
		kernel_min = np.zeros_like(kernel_guess)
		x0 = np.concatenate([[lambda_guess], kernel_guess, scale_guess])
		hyper_max = np.concatenate([[lambda_max], kernel_max, scale_max])
		hyper_min = np.concatenate([[lambda_min], kernel_min, scale_min])
		generic_kernel = matern52_kernel
		
	elif kernel=='poly2_matern32':
	
		scale_guess = np.array([(np.max(X[:,i]) - np.min(X[:,i])) for i in range(nparams)])
		scale_max = 1e2  * scale_guess
		scale_min = 1e-2 * scale_guess
		scale2_guess = np.array([(np.max(X[:,i]) - np.min(X[:,i])) for i in range(nparams)])
		scale2_max = 1e2  * scale2_guess
		scale2_min = 1e-2 * scale2_guess
		kernel_guess = np.array([(1./3.)*(np.var(Y) - lambda_guess),
								 (1./3.)*(np.var(Y) - lambda_guess),
								 (1./3.)*(np.var(Y) - lambda_guess)])
		kernel_max = 1.0 * np.ones_like(kernel_guess)
		kernel_min = 1e-3 * np.ones_like(kernel_guess)
		x0 = np.concatenate([[lambda_guess], kernel_guess, scale_guess, scale2_guess])
		hyper_max = np.concatenate([[lambda_max], kernel_max, scale_max, scale2_max])
		hyper_min = np.concatenate([[lambda_min], kernel_min, scale_min, scale2_min])
		generic_kernel = poly2_matern32_kernel
		
	elif kernel=='sqrexp_matern32':
		
		scale_guess = np.array([(np.max(X[:,i]) - np.min(X[:,i])) for i in range(nparams)])
		scale_max = 1e2  * scale_guess
		scale_min = 1e-2 * scale_guess
		scale2_guess = np.array([(np.max(X[:,i]) - np.min(X[:,i])) for i in range(nparams)])
		scale2_max = 1e2  * scale2_guess
		scale2_min = 1e-2 * scale2_guess
		kernel_guess = np.array([(1./3.)*(np.var(Y) - lambda_guess),
								 (1./3.)*(np.var(Y) - lambda_guess),
								 (1./3.)*(np.var(Y) - lambda_guess)])
		kernel_max = 1.0 * np.ones_like(kernel_guess)
		kernel_min = 1e-3 * np.ones_like(kernel_guess)
		x0 = np.concatenate([[lambda_guess], kernel_guess, scale_guess, scale2_guess])
		hyper_max = np.concatenate([[lambda_max], kernel_max, scale_max, scale2_max])
		hyper_min = np.concatenate([[lambda_min], kernel_min, scale_min, scale2_min])
		generic_kernel = sqrexp_matern32_kernel

	elif kernel=='sqrexp_matern52':
		
		scale_guess = np.array([(np.max(X[:,i]) - np.min(X[:,i])) for i in range(nparams)])
		scale_max = 1e2  * scale_guess
		scale_min = 1e-2 * scale_guess
		scale2_guess = np.array([(np.max(X[:,i]) - np.min(X[:,i])) for i in range(nparams)])
		scale2_max = 1e2  * scale2_guess
		scale2_min = 1e-2 * scale2_guess
		kernel_guess = np.array([(1./3.)*(np.var(Y) - lambda_guess),
								 (1./3.)*(np.var(Y) - lambda_guess),
								 (1./3.)*(np.var(Y) - lambda_guess)])
		kernel_max = 1.0 * np.ones_like(kernel_guess)
		kernel_min = 1e-3 * np.ones_like(kernel_guess)
		x0 = np.concatenate([[lambda_guess], kernel_guess, scale_guess, scale2_guess])
		hyper_max = np.concatenate([[lambda_max], kernel_max, scale_max, scale2_max])
		hyper_min = np.concatenate([[lambda_min], kernel_min, scale_min, scale2_min])
		generic_kernel = sqrexp_matern52_kernel

	elif kernel=='sqrexp':
		
		scale_guess = np.array([(np.max(X[:,i]) - np.min(X[:,i])) for i in range(nparams)])
		scale_max = 1e2  * scale_guess
		scale_min = 1e-2 * scale_guess

		kernel_guess = np.array([(1./3.)*(np.var(Y) - lambda_guess),
								 (1./3.)*(np.var(Y) - lambda_guess)])

		kernel_max = 1.0 * np.ones_like(kernel_guess)
		kernel_min = 1e-3 * np.ones_like(kernel_guess)

		x0 = np.concatenate([[lambda_guess], kernel_guess, scale_guess])
		hyper_max = np.concatenate([[lambda_max], kernel_max, scale_max])
		hyper_min = np.concatenate([[lambda_min], kernel_min, scale_min])
		generic_kernel = sqrexp_kernel

	else:
	
		print("Kernel is not valid or not specified!")
		exit(1)


	@jit
	def negative_log_likelihood(h):
		K = compute_kernel(X,h[1:],generic_kernel)
		c, rms_LOOE_norm,logp,err_vector = fit_data_with_kernel(Y,K,Yerr,h[0])
		return -logp


	@jit
	def leave_one_out_norm(h):
		K = compute_kernel(X,h[1:],generic_kernel)
		c, rms_LOOE_norm,logp,err_vector = fit_data_with_kernel(Y,K,Yerr,h[0])
		return rms_LOOE_norm


	## optimize hyperparameters

	print('\toptimizing hyperparameters...', end='', flush=True)
	start_time = time.time()
	
	outer_opt = negative_log_likelihood    
	if kernel=='polyharmonic' or kernel=='spline':
		outer_opt = leave_one_out_norm
	
	f = lambda h, grad: outer_opt(h)

	opt = nlopt.opt(nlopt.LN_BOBYQA, x0.shape[0]) # tends to work better for this problem
	opt.set_min_objective(f)
	opt.set_xtol_rel(1e-5)
	opt.set_ftol_rel(1e-5)
	opt.set_lower_bounds(hyper_min)
	opt.set_upper_bounds(hyper_max)
	xopt = opt.optimize(x0)
	best_h = (xopt[1:])
	best_lambda = (xopt[0])
	
	print(f'\tdone in {time.time() - start_time} seconds.')

	K = compute_kernel(X,best_h,generic_kernel)
	c, rms_LOOE_norm, logp, err_vector = fit_data_with_kernel(Y,K,Yerr,best_lambda)

	return c, rms_LOOE_norm, best_lambda, best_h, logp, err_vector, generic_kernel


@jit
def model_data(X_input, X_model, c, h, generic_kernel):

	Y_model = np.zeros(X_model.shape[0])
	for i in range(X_model.shape[0]):
		for j in range(c.shape[0]):
			Y_model[i] += generic_kernel(X_input[j,:], X_model[i,:], h) * c[j]
			
	return Y_model


def compute_labels(x):

	"""normalize inputs to a [0, 1] hyperrectangle."""
	
	labels = np.empty_like(x)
	for j in range(x.shape[1]):
		xmin = x[:,j].min()
		xmax = x[:,j].max()
		varrange = xmax - xmin
		labels[:,j] = (x[:,j] - xmin)/varrange
		
	return labels


def load_correlation_file(filename):

	table = np.loadtxt(filename,unpack=False)
	binmin, binmax, err_corr, corr = [table[:,i] for i in range(4)]
	return binmin, binmax, err_corr, corr


def load_scalar_file(filename):
	
	scalar = np.loadtxt(filename,unpack=False)
	return scalar


def read_data(sims_dir, redshift_dir, param_files, filename_ext=None, load_scalar=False):

	"""read parameters from *.template_param files, read in *.galaxy_bias files,
	read simulation /header files for cosmological params.
	return (matrix X of parameters, vector Y of observations) for each radial bin."""
	
	sims_path = Path(sims_dir)
	
	# each param_file should be of the form '*.template_param'
	def obs_filename(param_filename):
		return str(param_filename) + filename_ext
	
	if load_scalar == True:
		obs_vector = load_scalar_file(obs_filename(param_files[0]))
	else:
		binmin, binmax, obs_err, obs_vector = load_correlation_file(obs_filename(param_files[0]))

	if load_scalar == True:
		nbins = 1
	else:
		nbins = len(binmin)

	nobs = len(param_files)
	nparams = 17

	X = np.zeros((nobs, nparams))
	Y = np.zeros((nbins, nobs))
	Yerr = np.zeros((nbins, nobs))
	SimID = np.empty((nobs,), dtype=object)
	good_sample = np.empty(nobs, dtype=np.bool)
	good_sample[:] = True
	
	param_names_vector = np.asarray(['ngal','siglogM','M0_over_M1','M1_over_Mmin','alpha',
			'q_env','del_gamma','f_cen','A_conc','R_rescale',
			'redshift','sigma_8','H0','ombh2','omch2','w0','ns','sim_id'])

	for j, param_file in enumerate(param_files):
	
		myconfigparser = configparser.ConfigParser()
		myconfigparser.read(str(param_file))
		hod_params = myconfigparser['params']

		## read in simulation file
		
		simdir = hod_params['dir'].strip('"')
		subdir = sims_path / hod_params['dir'].strip('"') / redshift_dir
		header_file = str(subdir) + '/header'
		cf = config.AbacusConfigFile(header_file)
		redshift = cf.redshift
		sigma_8 = cf.sigma_8
		H0 = cf.H0
		ombh2 = cf.ombh2
		omch2 = cf.omch2
		w0 = cf.w0
		ns = cf.ns

		bias_file = obs_filename(param_file)
		if not Path(bias_file).exists():
			continue # ignore missing files

		if load_scalar == True:
			obs_vector = load_scalar_file(bias_file)
			obs_err = (obs_vector * 1e-2)	# arbitrary
		else:
			binmin, binmax, obs_err, obs_vector = load_correlation_file(bias_file)

		hod_vector = np.array( [ float(hod_params['ngal']),
								 float(hod_params['siglogM']),
								 float(hod_params['M0_over_M1']),
								 float(hod_params['M1_over_Mmin']),
								 float(hod_params['alpha']),
								 float(hod_params['q_env']),
								 float(hod_params['del_gamma']),
								 float(hod_params['f_cen']),
								 float(hod_params['A_conc']),
								 float(hod_params['R_rescale']) ])


		cosmo_vector = np.array([redshift, sigma_8, H0, ombh2, omch2, w0, ns])
		this_param_vector = np.concatenate([hod_vector, cosmo_vector])

		## reject extreme samples (*and* delete these rows from the arrays)
#		if (alpha > 1.8):  # most of the difficulty is predicting high alpha models
#		if (M1_over_Mmin < 8.0 and alpha < 0.7):
#			good_sample[j] = False

		assert(np.all(obs_err > 0.))

		X[j,:] = this_param_vector
		Y[:,j] = obs_vector
		Yerr[:,j] = obs_err
		SimID[j] = simdir


	## delete rows of X, Y, Yerr that have np.NaN entries
	
	mask = good_sample
	X = X[mask, :]
	Y = Y[:, mask]
	Yerr = Yerr[:, mask]
	SimID = SimID[mask]
	good_param_files = np.asarray(param_files)[mask]
	
	## delete columns of X that have zero variance (for numerical stability)

	idx = []
	for i in range(X.shape[1]):
		x = X[:,i]
		var = np.var(x)
		eps = 1e-15
		if var > eps:
			print(f"variance for parameter {i} [{param_names_vector[i]}] = {var}; range = [{np.min(x)}, {np.max(x)}]")
			idx.append(i)

	## remove zero-variance columns

	X = X[:,idx]
	param_names = param_names_vector[idx]

	print("nparams: {}".format(X.shape[1]))
	print("nobs: {}".format(X.shape[0]))

	if load_scalar == True:
		binmin = 0.
		binmax = 0.

	return X, Y, Yerr, SimID, binmin, binmax, param_names, good_param_files


def fit_data(sims_dir, redshift_dir, param_files, kernel,
			 output_emu_filename, obs_filename_ext, load_scalar=False):
			
	"""fit data with a linear least-squares model"""

	x, y_allbins, yerr_allbins, sim_ids, binmin, binmax, param_names, param_files_used = \
			read_data(sims_dir, redshift_dir, param_files, filename_ext=obs_filename_ext,
						load_scalar=load_scalar)

	binmed = 0.5*(binmin+binmax)
	
	print(f"binmin = {binmin}")
	print(f"binmax = {binmax}")

	# solve: min_c { |Y - Kc|^2 + Lambda |c|^2 }
	# solution: c = K^{-1} Y
	
	X = compute_labels(x) # can modify zero-point, take logarithms, etc.
	x0 = np.zeros((1,X.shape[1]))

	coefs = np.zeros((y_allbins.shape[0], X.shape[0]))
	kernel_hypers = []
	y_mean = np.zeros(y_allbins.shape[0])
	y_sigma = np.zeros(y_allbins.shape[0])
	sim_label = pd.Series(sim_ids, dtype="category")

	rms_looe = np.zeros(y_allbins.shape[0])
	rms_residuals = np.zeros(y_allbins.shape[0])
#	rms_kfold = np.zeros(y_allbins.shape[0])
	rms_simfold = np.zeros(y_allbins.shape[0])
	frac_rms_looe = np.zeros(y_allbins.shape[0])
	frac_rms_residuals = np.zeros(y_allbins.shape[0])
#	frac_rms_kfold = np.zeros(y_allbins.shape[0])
	frac_rms_simfold = np.zeros(y_allbins.shape[0])
	looe_err_vectors = np.zeros(y_allbins.shape)
	simfold_err_vectors = np.zeros(y_allbins.shape)

	for j in range(y_allbins.shape[0]):
	
		y = y_allbins[j,:]
		yerr = yerr_allbins[j,:]
		y0 = np.mean(y)
		sigma_y = np.sqrt(np.var(y))
		y_mean[j] = y0
		y_sigma[j] = sigma_y
		Y = np.matrix((y-y0)/sigma_y).T # column vector
		Yerr = np.matrix(yerr/sigma_y).T

		c, rms_LOOE_norm, best_lambda, best_h, logp, err_vector, generic_kernel = \
										fit_hyperparameters(Y,X,Yerr,kernel=kernel)
		coefs[j,:] = np.squeeze(c)
		kernel_hypers.append(np.concatenate([np.array([best_lambda]), best_h]))
		looe_err_vectors[j,:] = err_vector
		
		y_model_x = sigma_y*model_data(X, X, c, best_h, generic_kernel) + y0
		rms_model_residuals = np.sqrt(np.mean((y_model_x - y)**2))
		
		rms_looe[j] = sigma_y*rms_LOOE_norm
		rms_residuals[j] = rms_model_residuals
		frac_rms_looe[j] = sigma_y*rms_LOOE_norm / y_mean[j]
		frac_rms_residuals[j] = rms_model_residuals / y_mean[j]


		## at fixed hyperparameters, compute Nsim-fold cross-validation

		norm_simfold_cv = 0.
		simfolds = len(sim_label.cat.categories)

		for k in range(simfolds):

			## select fold k of data

			Nobs = X.shape[0]
			this_sim = sim_label.cat.categories[k]
			mask = (sim_label == this_sim)
			
			X_k_input   	= X[~mask,:]
			y_k_input   	= y[~mask]
			yerr_k_input	= yerr[~mask]

			X_k_holdout 	= X[mask,:]
			y_k_holdout 	= y[mask]
			yerr_k_holdout  = yerr[mask]

			## fit model (using fixed hyperparameters)
			
			K_k = compute_kernel(X_k_input, best_h, generic_kernel)
			Y_k = np.matrix((y_k_input - y0)/sigma_y).T
			Yerr_k = np.matrix(yerr_k_input/sigma_y).T
			
			c_k, rms_LOOE_norm_k, logp_k, err_vec_k = fit_data_with_kernel(Y_k,K_k,Yerr_k,best_lambda)
			y_model_holdout = y0 + sigma_y*model_data(X_k_input, X_k_holdout,
													  c_k, best_h, generic_kernel)
			simfold_err_vectors[j, mask] = (y_model_holdout - y_k_holdout)
			norm_simfold_cv += np.mean((y_model_holdout - y_k_holdout)**2) / simfolds

		rms_simfold_cv = np.sqrt(norm_simfold_cv)
		rms_simfold[j] = rms_simfold_cv
		frac_rms_simfold[j] = rms_simfold_cv / y_mean[j]

		print(f'[bin {j}] frac rms model residuals = {frac_rms_residuals[j]}')
		print(f'[bin {j}] frac rms leave-one-out error: {frac_rms_looe[j]}')
		print(f'[bin {j}] frac rms N_sim({simfolds})-fold CV error: {frac_rms_simfold[j]}')
		print(f'[bin {j}] best L2-regularization hyperparameter = {best_lambda}')
		print(f'[bin {j}] log pseudo-likelihood: {logp}')
		print(f'[bin {j}] best scale hyperparameters = {best_h}')
		print('')


	## save emulator
	
	with h5py.File(output_emu_filename,'w') as f:
	
		utf8_param_files = [s.encode("UTF-8", "ignore") for s in param_files_used]
		utf8_param_names = [s.encode("UTF-8", "ignore") for s in param_names]
	
		f.create_dataset("this_filename", data=output_emu_filename)
		f.create_dataset("simulations_dir", data=sims_dir)
		f.create_dataset("redshift_dir", data=redshift_dir)
		f.create_dataset("param_files_inputs", data=utf8_param_files)
		f.create_dataset("filename_extension_inputs", data=obs_filename_ext)
		f.create_dataset("param_names_inputs", data=utf8_param_names)
		f.create_dataset("rbins_min", data=binmin)
		f.create_dataset("rbins_max", data=binmax)
		
		f.create_dataset("raw_training_inputs", data=x)
		f.create_dataset("raw_training_outputs", data=y_allbins)
		f.create_dataset("err_raw_training_outputs", data=yerr_allbins)

		f.create_dataset("normalized_training_inputs", data=X)
		f.create_dataset("mean_training_outputs", data=y_mean)
		f.create_dataset("stdev_training_outputs", data=y_sigma)
		
		f.create_dataset("best_fit_coefs", data=coefs)
		f.create_dataset("looe_err_vectors", data=looe_err_vectors)
		f.create_dataset("simfold_err_vectors", data=simfold_err_vectors)
		f.create_dataset("rms_loo_cv_err", data=rms_looe)
		f.create_dataset("rms_simfold_cv_err", data=rms_simfold)
		f.create_dataset("rms_residuals", data=rms_residuals)
		
		f.create_dataset("gp_kernel_hyperparameters", data=np.asarray(kernel_hypers))
		f.create_dataset("gp_kernel_name", data=kernel)
		f.create_dataset("gp_kernel_sourcecode", data=inspect.getsource(generic_kernel))



if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	
	parser.add_argument('--output_emu_filename', required=True)
	parser.add_argument('--sims_dir', default='./AbacusCosmos/AbacusCosmos_720box/Rockstar')
	parser.add_argument('--redshift_dir', default='z0.300')
	parser.add_argument('param_files', nargs='*')
	
	parser.add_argument('--kernel', default='sqrexp')
	parser.add_argument('--obs_ext', default='.ratio_wp.txt')
	
	parser.add_argument('--load-scalar', default=False, action='store_true')

	args = parser.parse_args()
	
	fit_data(args.sims_dir, args.redshift_dir, args.param_files, args.kernel,
			 args.output_emu_filename, args.obs_ext, load_scalar=args.load_scalar)
