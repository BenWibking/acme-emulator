import h5py
import configparser
import nlopt
import mpmath
import mpi4py
from pathlib import Path
import scipy.stats

import numpy as np
import scipy.linalg

from predict_emulator import get_emulate_fun, compute_analytic_wp


def exec_mpi(func, *args):
	import sys
	if 'mpi4py' not in sys.modules:
		func(*args)
	else:
		import mpi4py.MPI
		comm = mpi4py.MPI.COMM_WORLD
		rank = comm.Get_rank()
		if rank == 0:
			func(*args)

def print_mpi(*args):
	exec_mpi(print, *args)

def savetxt_mpi(*args):
	exec_mpi(np.savetxt, *args)


def sherman_morrison_singular_inv(A, u):

	"""compute the inverse covariance matrix that projects out the data vector v.
	   For lnL_correction, see eq. (10) arXiv:astro-ph/0112114v2."""

	Ainv = np.matrix( np.linalg.inv(A) )
	v = np.matrix(u)

	Cinv = 	Ainv - (Ainv @ v.T @ v @ Ainv) / (v @ Ainv @ v.T)
	(sign, logdet) = np.linalg.slogdet(1.0 + v @ Ainv @ v.T)
	lnL_correction = -0.5*logdet

	return Cinv, lnL_correction


if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser()

	parser.add_argument('wp_obs_file')
	parser.add_argument('input_wp_emu_filename')
	parser.add_argument('wp_cov_file')

	parser.add_argument('ds_obs_file')
	parser.add_argument('input_ds_emu_filename')
	parser.add_argument('ds_cov_file')

	parser.add_argument('fiducial_param_file')
	parser.add_argument('param_sampling_file')

	parser.add_argument('fit_wp_file')
	parser.add_argument('fit_ds_file')
	parser.add_argument('fit_param_file')

	parser.add_argument('--multinest', default=None)

	parser.add_argument('--mcmc', default=False, action='store_true')
	parser.add_argument('--mcmc-ppd')
	parser.add_argument('--mcmc-chain')

	parser.add_argument('--wp-correction', default=None)	# from phases ensemble
	parser.add_argument('--ds-correction', default=None)	# from phases ensemble
	parser.add_argument('--rsd-correction', default=None)

	parser.add_argument('--cov-inv-output')
	parser.add_argument('--cov-output')

	args = parser.parse_args()


	## fiducial parameters
	parser = configparser.ConfigParser()
	parser.read(args.fiducial_param_file)
	fiducial_params = parser['params']

	parser_range = configparser.ConfigParser()
	parser_range.read(args.param_sampling_file)
	min_params = parser_range['params_min']
	max_params = parser_range['params_max']
	gaussian_priors = parser_range['gaussian_priors']
	analysis_settings = parser_range['settings']

	use_pointmass = analysis_settings['use_pointmass'].upper()
	print(f"use_pointmass = {use_pointmass}")

	ds_calibration_prior_sigma = float(gaussian_priors['ds_calibration_prior'])

	ngal_fiducial = float(fiducial_params['ngal'])
	siglogM_fiducial = float(fiducial_params['siglogM'])
	M0_ratio_fiducial = float(fiducial_params['M0_over_M1'])
	M1_ratio_fiducial = float(fiducial_params['M1_over_Mmin'])
	alpha_fiducial = float(fiducial_params['alpha'])
	fcen_fiducial = float(fiducial_params['f_cen'])
	Aconc_fiducial = float(fiducial_params['A_conc'])
	Rrescale_fiducial = float(fiducial_params['R_rescale'])

	omegam_fiducial = float(fiducial_params['omega_m'])
	sigma8_fiducial = float(fiducial_params['sigma_8'])
	H0_fiducial = float(fiducial_params['H0'])
	ns_fiducial = float(fiducial_params['ns'])
	omegab_fiducial = float(fiducial_params['omega_b'])
	w0_fiducial = float(fiducial_params['w0'])
	redshift_fiducial = float(fiducial_params['redshift'])
	ds_calibration_fiducial = 1.0

	fcen_fiducial = 1.0

	ngal_min = float(min_params['ngal'])
	siglogM_min = float(min_params['siglogM'])
	M0_ratio_min = float(min_params['M0_over_M1'])
	M1_ratio_min = float(min_params['M1_over_Mmin'])
	alpha_min = float(min_params['alpha'])
#	fcen_min = float(min_params['f_cen'])
	Aconc_min = float(min_params['A_conc'])
	Rrescale_min = float(min_params['R_rescale'])

	omegam_min = float(min_params['omega_m'])
	sigma8_min = float(min_params['sigma_8'])
	H0_min = float(min_params['H0'])
	ns_min = float(min_params['ns'])
	omegab_min = float(min_params['omega_b'])
	w0_min = float(min_params['w0'])
	redshift_min = float(min_params['redshift'])
	ds_calibration_min = 0.8

	ngal_max = float(max_params['ngal'])
	siglogM_max = float(max_params['siglogM'])
	M0_ratio_max = float(max_params['M0_over_M1'])
	M1_ratio_max = float(max_params['M1_over_Mmin'])
	alpha_max = float(max_params['alpha'])
#	fcen_max = float(max_params['f_cen'])
	Aconc_max = float(max_params['A_conc'])
	Rrescale_max = float(max_params['R_rescale'])

	omegam_max = float(max_params['omega_m'])
	sigma8_max = float(max_params['sigma_8'])
	H0_max = float(max_params['H0'])
	ns_max = float(max_params['ns'])
	omegab_max = float(max_params['omega_b'])
	w0_max = float(max_params['w0'])
	redshift_max = float(max_params['redshift'])
	ds_calibration_max = 1.2

	assert( redshift_min == redshift_max )
	redshift_fiducial = redshift_min


	## read in observed wp

	print_mpi("reading in observed wp and DS...")

	obs_binmin, obs_binmax, mock_wp_err, mock_wp = np.loadtxt(args.wp_obs_file, unpack=True)
	ds_binmin,  ds_binmax,  mock_ds_err, mock_ds = np.loadtxt(args.ds_obs_file, unpack=True)

	assert( np.allclose( obs_binmin, ds_binmin ) )
	assert( np.allclose( obs_binmax, ds_binmax ) )

	scale_min = 0.6 # Mpc/h (determined by fiber collisions)
#	scale_min = 2.0 # Mpc/h
	scale_max = np.inf
	scale_mask = np.logical_and((obs_binmin >= scale_min), (obs_binmax <= scale_max))

	rp_vector = ( 0.5*(obs_binmin + obs_binmax) )[scale_mask]
	mock_datavector = np.concatenate([mock_wp[scale_mask], mock_ds[scale_mask]])

	print_mpi(f"obs_binmin: {obs_binmin}")
	print_mpi(f"obs_binmax: {obs_binmax}")


	## read wp ratio correction

	if args.wp_correction is not None and Path(args.wp_correction).exists():
		corr_binmin, corr_binmax, _, wp_correction = np.loadtxt(args.wp_correction,
																	  unpack=True)
		assert( np.allclose(corr_binmin, obs_binmin) )
		assert( np.allclose(corr_binmax, obs_binmax) )
	else:
		wp_correction = np.zeros_like(mock_wp)


	print_mpi(f"wp correction (multiplicative) = {wp_correction}")


	## read DS correction (multiplicative)

	if args.ds_correction is not None and Path(args.ds_correction).exists():
		dscorr_binmin, dscorr_binmax, _, ds_correction = np.loadtxt(args.ds_correction,
																	unpack=True)
		assert( np.allclose(dscorr_binmin, obs_binmin) )
		assert( np.allclose(dscorr_binmax, obs_binmax) )
	else:
		ds_correction = np.zeros_like(mock_ds)

	print_mpi(f"ds correction (multiplicative) = {ds_correction}")


	## read RSD correction

	if args.rsd_correction is not None and Path(args.rsd_correction).exists():
		rsd_binmin, rsd_binmax, _, rsd_correction = np.loadtxt(args.rsd_correction,unpack=True)
		assert( np.allclose(rsd_binmin, obs_binmin) )
		assert( np.allclose(rsd_binmax, obs_binmax) )
	else:
		rsd_correction = np.ones_like(mock_wp)

	print_mpi(f"rsd correction (multiplicative) = {rsd_correction}")
	print_mpi(f"")


	## read in covariance for wp

	wp_cov = np.loadtxt(args.wp_cov_file)
	wp_cov = wp_cov[scale_mask,:][:,scale_mask]


	## rescale wp covariance for appropriate volume
	# [effective volume computed for LOWZ (but is actually NGC only!)]
	R_survey = 913.26	# Mpc/h

	box_length = 1100.	# Mpc/h
	Nbootstrap = 20*25						## TODO: make this an input parameter!
	Nmocks = Nbootstrap						#  these happen to be the same
	vol_subvolume = (box_length)**3 / 25
	vol_survey = R_survey**3
	wp_cov *= Nbootstrap					## need to rescale by number of bootstrap samples
	wp_cov *= (vol_subvolume / vol_survey)	## need to rescale by inv. vol of survey


	## read in DS covariance [must be already normalized!]

	ds_cov = np.loadtxt(args.ds_cov_file)
	ds_cov = ds_cov[scale_mask,:][:,scale_mask]


	## combine covariance matrices, neglect cross-covariance between wp, DS

	cov = np.block([
					[wp_cov,				np.zeros_like(wp_cov)],
					[np.zeros_like(wp_cov),	ds_cov				 ] ])


	## compute inverse covariance matrix (de-noise the matrix before, if needed)
	## [use the Sherman-Morrison-Woodbury formula to compute the subspace inverse]

	# threshold = 2.0/np.sqrt(Nmocks)
	# U, s, V = scipy.linalg.svd(cov)
	# s[s < threshold] = 0.
	# sinv = np.zeros_like(s)
	# sinv[s >= threshold] = 1.0/s[s >= threshold]
	# cov_inv = np.dot(V.T, np.dot(np.diag(sinv), U.T))

	if use_pointmass == "TRUE":
		print(f"using point mass term.")
		point_mass_term = (scale_min / rp_vector)**2
		point_mass_datavector = np.concatenate([np.zeros(wp_cov.shape[0]), point_mass_term])
		cov_inv, lnL_correction = sherman_morrison_singular_inv(cov, point_mass_datavector.T)
		cov_inv = np.array(cov_inv)
		
	else:
		print(f"NOT using point mass term.")
		cov_inv = np.linalg.inv(cov)	# don't modify
		lnL_correction = 0.	

	if args.cov_inv_output is not None:
		np.savetxt(args.cov_inv_output, cov_inv)
	if args.cov_output is not None:
		np.savetxt(args.cov_output, cov)


	## compute log det(cov) [necessary for normalizing log-likelihood]

	(sign, logdetcov) = np.linalg.slogdet(cov)

	Nobs = len(mock_datavector)
	lnL_norm_uncorrected = -0.5*Nobs*np.log(2.0*np.pi) - 0.5*logdetcov
	lnL_norm = lnL_norm_uncorrected + lnL_correction

	print_mpi(f"log det(cov) = {logdetcov}")
	print_mpi(f"lnL normalization = {lnL_norm} = {lnL_norm_uncorrected} + {lnL_correction}")


	## print uncertainties

	wp_err = np.sqrt( np.diag(wp_cov) ) / mock_wp[scale_mask]
	wp_err_jackknife = mock_wp_err[scale_mask] / mock_wp[scale_mask]

	print_mpi(f"wp fractional error (w/ noise) = {wp_err}")
	print_mpi(f"wp fractional error (jackknife) = {wp_err_jackknife}")

	ds_err = np.sqrt( np.diag(ds_cov) ) / mock_ds[scale_mask]
	ds_err_jackknife = mock_ds_err[scale_mask] / mock_ds[scale_mask]

	print_mpi(f"ds fractional error (w/ noise) = {ds_err}")
	print_mpi(f"ds fractional error (jackknife) = {ds_err_jackknife}")


	## fit mock observations with HOD model + wCDM cosmological parameters

	wp_emulate_fun, emu_binmin, emu_binmax = get_emulate_fun(args.input_wp_emu_filename)
	try:
		assert( np.allclose(emu_binmin, obs_binmin) )
		assert( np.allclose(emu_binmax, obs_binmax) )
	except:
		print_mpi(f"emu_binmin = {emu_binmin}")
		exit(1)

	ds_emulate_fun, dsemu_binmin, dsemu_binmax = get_emulate_fun(args.input_ds_emu_filename)
	try:
		assert( np.allclose(dsemu_binmin, obs_binmin) )
		assert( np.allclose(dsemu_binmax, obs_binmax) )
	except:
		print_mpi(f"dsemu_binmin = {dsemu_binmin}")
		exit(1)


	def this_halo_model(params, print=False):

		ngal 	 	 = params[0]
		M0_over_M1	 = params[1]
		M1_over_Mmin = params[2]
		siglogM 	 = params[3]
		alpha		 = params[4]
		A_conc		 = params[5]
		R_rescale	 = params[6]
		f_cen		 = fcen_fiducial

		omega_m 		 = params[7]
		sigma_8			 = params[8]
		H0				 = params[9]
		ns				 = params[10]
		omegab			 = params[11]
		w0				 = params[12]
		redshift		 = redshift_fiducial
		ds_calibration	 = params[13]

		omegac = omega_m - omegab
		h = H0/100.
		ombh2 = omegab*h**2
		omch2 = omegac*h**2

		if print == True:
			print_mpi(f"\t{ngal*1.0e4:.5f} {siglogM:.5f} " + \
					  f"{M1_over_Mmin:.5f} {M0_over_M1:.5f} " + \
					  f"{alpha:.5f} {A_conc:.5f} {R_rescale:.5f} {omega_m:.5f} {sigma_8:.5f} " + \
					  f"{ds_calibration:.2f}")

		this_analytic_wp = compute_analytic_wp(omega_m, omegab, H0, ns,
											   w0, sigma_8,
											   redshift,
											   ngal, siglogM,
											   M1_over_Mmin, M0_over_M1, alpha, f_cen,
											   wp_binmin=obs_binmin,
											   wp_binmax=obs_binmax)


		## compute emulator correction

		input_parameters = np.array([ngal, siglogM, M0_over_M1, M1_over_Mmin, alpha,
									 A_conc, R_rescale,
								 	 sigma_8, H0, ombh2, omch2, w0, ns])

		predicted_wp_ratio_uncorrected = wp_emulate_fun(input_parameters)
		predicted_ds_uncorrected = ds_emulate_fun(input_parameters)


		## add correction from additional phases

		predicted_wp_norsd = (predicted_wp_ratio_uncorrected * this_analytic_wp) * wp_correction
		predicted_ds = predicted_ds_uncorrected * ds_correction

		this_wp = (predicted_wp_norsd * rsd_correction)
		this_ds = (predicted_ds) * ds_calibration
		
#		if print:
#			print_mpi(f"\temu wp ratio = {predicted_wp_ratio_uncorrected}")
#			print_mpi(f"\tanalytic wp = {this_analytic_wp}")
#			print_mpi(f"\twp_correction = {wp_correction}")
#			print_mpi(f"\twp = {this_wp}")
#			print_mpi(f"")

		return this_wp, this_ds


	## params: ngal, M0_over_M1, M1_over_Mmin, siglogM, alpha, A_conc, R_rescale
	##		   omega_m, sigma_8, H0, ns, omegab, w0

	params_guess = np.array([ngal_fiducial,
							 M0_ratio_fiducial,
							 M1_ratio_fiducial,
							 siglogM_fiducial,
							 alpha_fiducial,
							 Aconc_fiducial,
							 Rrescale_fiducial,
							 omegam_fiducial,
							 sigma8_fiducial,
							 H0_fiducial,
							 ns_fiducial,
							 omegab_fiducial,
							 w0_fiducial,
							 ds_calibration_fiducial])

	params_min   = np.array([ngal_min,
							 M0_ratio_min,
							 M1_ratio_min,
							 siglogM_min,
							 alpha_min,
							 Aconc_min,
							 Rrescale_min,
							 omegam_min,
							 sigma8_min,
							 H0_min,
							 ns_min,
							 omegab_min,
							 w0_min,
							 ds_calibration_min])

	params_max   = np.array([ngal_max,
							 M0_ratio_max,
							 M1_ratio_max,
							 siglogM_max,
							 alpha_max,
							 Aconc_max,
							 Rrescale_max,
							 omegam_max,
							 sigma8_max,
							 H0_max,
							 ns_max,
							 omegab_max,
							 w0_max,
							 ds_calibration_max])


	def ln_L(params, disable_boundscheck=False, print=False):

		if disable_boundscheck == True \
			or (np.all(params >= params_min) and np.all(params <= params_max)):

			this_wp, this_ds = this_halo_model(params, print=print)
			this_datavector = np.concatenate([this_wp[scale_mask], this_ds[scale_mask]])
			dy = (this_datavector - mock_datavector)

			chisq = np.dot(dy, np.dot(cov_inv, dy))
			log_likelihood = -0.5*chisq + lnL_norm

			blob = (this_wp, this_ds, chisq)	 # save the posterior predictive distribution
			return log_likelihood, blob

		else:

			return -np.inf, None


	param_range = params_max - params_min
	fixed_params = (param_range == 0.0)
	param_range[fixed_params] = 1.0

	def f(x, grad):
		params = params_guess.copy()
		params[np.logical_not(fixed_params)] = x
		log_likelihood, blob = ln_L(params, print=True)
		return log_likelihood*(-1.0)

	def transform(params):
		unitized = (params - params_min) / param_range
		unitized[fixed_params] = 0.
		return unitized

	def untransform(unitized):
		params = unitized*param_range + params_min
		params[fixed_params] = params_min[fixed_params]
		return params

	Nobs = len(mock_datavector)
	Nparams_fitted = np.count_nonzero(~fixed_params)
	dof = Nobs - Nparams_fitted
	print_mpi(f"dof = {dof} ({Nobs} observations for {Nparams_fitted} fitted parameters)")


	if not Path(args.fit_param_file).exists():

		print_mpi("fitting wp...")
	
		opt = nlopt.opt(nlopt.LN_BOBYQA, np.count_nonzero(np.logical_not(fixed_params)))
		opt.set_min_objective(f)
		opt.set_xtol_rel(1e-3)
		opt.set_ftol_rel(1e-4)
		opt.set_lower_bounds(params_min[np.logical_not(fixed_params)])
		opt.set_upper_bounds(params_max[np.logical_not(fixed_params)])
		best_fit_params = opt.optimize(params_guess[np.logical_not(fixed_params)])
		
		maximum_likelihood_params = params_guess.copy()
		maximum_likelihood_params[np.logical_not(fixed_params)] = best_fit_params
	
		print(f"best fit params: {best_fit_params}")
		

		## output best-fit parameters to file

		config = configparser.ConfigParser()
		config['params'] = {}
		params = config['params']

		params['ngal'] 	 	   = str( maximum_likelihood_params[0] )
		params['M0_over_M1']   = str( maximum_likelihood_params[1] )
		params['M1_over_Mmin'] = str( maximum_likelihood_params[2] )
		params['siglogM'] 	   = str( maximum_likelihood_params[3] )
		params['alpha']		   = str( maximum_likelihood_params[4] )
		params['A_conc']	   = str( maximum_likelihood_params[5] )
		params['R_rescale']	   = str( maximum_likelihood_params[6] )
		params['f_cen']		   = str( fcen_fiducial )

		params['omega_m'] 	 = str( maximum_likelihood_params[7] )
		params['sigma_8']	 = str( maximum_likelihood_params[8] )
		params['H0']		 = str( maximum_likelihood_params[9] )
		params['ns']		 = str( maximum_likelihood_params[10] )
		params['omega_b']	 = str( maximum_likelihood_params[11] )
		params['w0']		 = str( maximum_likelihood_params[12] )
		params['redshift']	 = str( redshift_fiducial )
		params['ds_calibration'] = str( maximum_likelihood_params[13] )

		def write_config():
			with open(args.fit_param_file, 'w') as configfile:
				config.write(configfile)

		exec_mpi(write_config)
		print_mpi(f"Wrote best-fit parameters to: {args.fit_param_file}.")

	else:

		## read best-fit parameters

		print_mpi(f"reading best-fit parameters from: {args.fit_param_file}")

		config = configparser.ConfigParser()
		config.read(args.fit_param_file)
		params = config['params']

		maximum_likelihood_params = np.zeros_like(params_guess)
		maximum_likelihood_params[0] = float(params['ngal'])
		maximum_likelihood_params[1] = float(params['M0_over_M1'])
		maximum_likelihood_params[2] = float(params['M1_over_Mmin'])
		maximum_likelihood_params[3] = float(params['siglogM'])
		maximum_likelihood_params[4] = float(params['alpha'])
		maximum_likelihood_params[5] = float(params['A_conc'])
		maximum_likelihood_params[6] = float(params['R_rescale'])
		maximum_likelihood_params[7] = float(params['omega_m'])
		maximum_likelihood_params[8] = float(params['sigma_8'])
		maximum_likelihood_params[9] = float(params['H0'])
		maximum_likelihood_params[10] = float(params['ns'])
		maximum_likelihood_params[11] = float(params['omega_b'])
		maximum_likelihood_params[12] = float(params['w0'])
		maximum_likelihood_params[13] = float(params['ds_calibration'])


	## compute best-fit diagnostics

	log_likelihood, (fit_wp, fit_ds, fit_chisq) = ln_L(maximum_likelihood_params)
	chisq_dof = fit_chisq / dof

	print_mpi("")
	print_mpi(f"best-fit log likelihood: {log_likelihood:.5f}")
	print_mpi(f"best-fit chi^2 = {fit_chisq:.5f} on {dof} d.o.f = {chisq_dof:.5f}")

	chisq_pvalue = float(mpmath.gammainc(dof/2.0, a=fit_chisq/2.0, regularized=True))

	print_mpi(f"\tp-value = {chisq_pvalue:.5f}")

	savetxt_mpi(args.fit_wp_file, np.c_[obs_binmin, obs_binmax,
										np.zeros_like(fit_wp), fit_wp])
	savetxt_mpi(args.fit_ds_file, np.c_[obs_binmin, obs_binmax,
										np.zeros_like(fit_ds), fit_ds])

	print_mpi(f"Wrote best-fit wp model to: {args.fit_wp_file}")
	print_mpi(f"Wrote best-fit DS model to: {args.fit_ds_file}")


	if args.multinest is not None:

		## run MultiNest (use `mpiexec -n 4 python ...` to run with MPI)

		import pymultinest
		
		assert (ds_calibration_prior_sigma >= 0.0)
		ds_calibration_prior = scipy.stats.norm(loc=1.0, scale=ds_calibration_prior_sigma)

		def prior_multinest(cube, ndim, nparams):

			"""map unit cube to prior of the problem space."""

			for i in range(ndim-1):
				cube[i] = cube[i]*(params_max[i] - params_min[i]) + params_min[i]

			if ds_calibration_prior_sigma > 0.0:
				cube[ndim-1] = ds_calibration_prior.ppf(cube[ndim-1])
			else:
				cube[ndim-1] = 1.0

		def lnL_multinest(params, ndim, nparams):

			"""return log likelihood at params 'params'."""

			real_params = params[0:ndim]
			loglike, (wp, ds, chisq) = ln_L(real_params, disable_boundscheck=True)

			## save posterior predictive distribution (PPD)

			params[nparams-1] = chisq

			for i in range(len(wp)):
				params[ndim+i] = wp[i]
			for i in range(len(ds)):
				params[ndim+len(wp)+i] = ds[i]

			return loglike


		ndims = len(params_min)
		nparams = ndims + len(mock_wp) + len(mock_ds) + 1

		pymultinest.run(lnL_multinest, prior_multinest, ndims,
						n_params = nparams,
						n_iter_before_update=10,
						importance_nested_sampling=False,
						n_live_points=800,
						sampling_efficiency=0.8,
						outputfiles_basename=args.multinest + '/1-',
						resume=True,
						verbose=True)

	elif args.mcmc == True:

		## run Markov Chain Monte Carlo, initialized at ML parameters

		import emcee
		nwalkers = 400
		ndim = maximum_likelihood_params.shape[0]
		p0 = [untransform(np.random.normal(loc=transform(maximum_likelihood_params), \
				scale=0.05)) for x in range(nwalkers)]

		sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_L, threads=8)


		## initialize file to save the posterior predictive distribution (PPD)

		ppdf = h5py.File(args.mcmc_ppd, 'w', libver='latest')

		empty_ln_l, empty_blob = ln_L(maximum_likelihood_params) # get dtypes automatically
		empty_wp, empty_chisq = empty_blob

		def create_dataset(name, init_data):
			data = np.array([init_data,])
			if not np.isscalar(init_data):
				chunksize = (100, init_data.shape[0])
				maxshape = (None, init_data.shape[0])
			else:
				chunksize = (100,)
				maxshape = (None,)
			return ppdf.create_dataset(name, chunks=chunksize, maxshape=maxshape, data=data)

		wp_dset = create_dataset("wp", empty_wp)
		lnL_dset = create_dataset("lnL", empty_ln_l)

		ppdf.swmr_mode = True

		def append_dataset(dset, item):
			if np.isscalar(item):
				new_shape = (len(dset) + 1,)
			else:
				new_shape = (len(dset) + 1, item.shape[0])
			dset.resize(new_shape)
			dset[-1] = item

		def flush_datasets():
			wp_dset.flush()
			lnL_dset.flush()


		## run Markov chain sampler

		for position, lnprob, rngstate, blobs in sampler.sample(p0,
																iterations=500,
																storechain=False):

			print_mpi(f"writing to {args.mcmc_chain}...")
			print_mpi(f"writing to {args.mcmc_ppd}...")

			f = open(args.mcmc_chain, "a")
			for k in range(position.shape[0]):
				this_blob = blobs[k]
				this_lnL = lnprob[k]

				if this_blob is not None:		# out of bounds if == None
					this_wp, this_chisq = this_blob
					f.write("{} {}\n".format(k, " ".join(map(str,position[k]))))
					append_dataset(wp_dset, this_wp)
					append_dataset(lnL_dset, this_lnL)
					flush_datasets()

			f.close()

		ppdf.close()

		print_mpi(f"finished running Markov chain.")
