#!/usr/bin/env python

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from predict_emulator import get_emulate_fun, compute_analytic_wp


if __name__ == '__main__':
	
	"""compute the emulator predictions across the range of emulated parameters. plot it."""

	parser = argparse.ArgumentParser()
	
	parser.add_argument('input_emu_filename')
	parser.add_argument('fiducial_params')
	parser.add_argument('param_range')
	parser.add_argument('output_plot')
	
	args = parser.parse_args()
	

	## read fiducial input parameters
	
	import configparser
	config = configparser.ConfigParser()
	config.read(args.fiducial_params)
	params = config['params']
	
	ngal = float(params['ngal'])
	M0_ratio = float(params['M0_over_M1'])
	M1_ratio = float(params['M1_over_Mmin'])
	siglogM = float(params['siglogM'])
	alpha 	= float(params['alpha'])
	Aconc	= float(params['A_conc'])
	Rrescale= float(params['R_rescale'])
	omegam = float(params['omega_m'])
	sigma8 = float(params['sigma_8'])
	H0 = float(params['H0'])
	ns = float(params['ns'])
	omegab = float(params['omega_b'])
	w0 = float(params['w0'])

	## read parameter ranges

	config = configparser.ConfigParser()
	config.read(args.param_range)
	min_params = config['params_min']
	max_params = config['params_max']

	ngal_min = float(min_params['ngal'])
	siglogM_min = float(min_params['siglogM'])
	M0_ratio_min = float(min_params['M0_over_M1'])
	M1_ratio_min = float(min_params['M1_over_Mmin'])
	alpha_min = float(min_params['alpha'])
	Aconc_min = float(min_params['A_conc'])
	Rrescale_min = float(min_params['R_rescale'])
	omegam_min = float(min_params['omega_m'])
	sigma8_min = float(min_params['sigma_8'])
	H0_min = float(min_params['H0'])
	ns_min = float(min_params['ns'])
	omegab_min = float(min_params['omega_b'])
	w0_min = float(min_params['w0'])

	ngal_max = float(max_params['ngal'])
	siglogM_max = float(max_params['siglogM'])
	M0_ratio_max = float(max_params['M0_over_M1'])
	M1_ratio_max = float(max_params['M1_over_Mmin'])
	alpha_max = float(max_params['alpha'])
	Aconc_max = float(max_params['A_conc'])
	Rrescale_max = float(max_params['R_rescale'])
	omegam_max = float(max_params['omega_m'])
	sigma8_max = float(max_params['sigma_8'])
	H0_max = float(max_params['H0'])
	ns_max = float(max_params['ns'])
	omegab_max = float(max_params['omega_b'])
	w0_max = float(max_params['w0'])

	params_fiducial = np.array([ngal,
							 M0_ratio,
							 M1_ratio,
							 siglogM,
							 alpha,
							 Aconc,
							 Rrescale,
							 omegam,
							 sigma8,
							 H0,
							 ns,
							 omegab,
							 w0])
							 							
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
							 w0_min])

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
							 w0_max])
							 
							 
	## compute emulated ratio

	emulate_fun, rp_binmin, rp_binmax = get_emulate_fun(args.input_emu_filename)
	rp_midbin = 0.5*(rp_binmin + rp_binmax)

	param_names = ["ngal", "M0_over_M1", "M1_over_Mmin", "siglogM", "alpha", "Aconc", "Rrescale",
		   		   "omegam", "sigma8", "H0", "ns", "omegab", "w0"]
		   		   
	labels=[r"n_{g} \times 10^{4}",
			r"M_0 / M_1",
			r"M_1 / M_{min}",
			r"\sigma_{\log M}",
			r"\alpha",
			r"A_{conc}",
			r"R_{rescale}",
			r"\Omega_m",
			r"\sigma_8",
			r"H_0",
			r"n_s",
			r"\Omega_b",
			r"w_0"]
	
	latex_labels = [r"$" + label + r"$" for label in labels]
	
	def emulate(params):

		ngal 	 	 = params[0]
		M0_over_M1	 = params[1]
		M1_over_Mmin = params[2]
		siglogM 	 = params[3]
		alpha		 = params[4]
		A_conc		 = params[5]
		R_rescale	 = params[6]

		omega_m 		 = params[7]
		sigma_8			 = params[8]
		H0				 = params[9]
		ns				 = params[10]
		omegab			 = params[11]
		w0				 = params[12]

		redshift = 0.300
		f_cen = 1.0

		omegac = omega_m - omegab
		h = H0/100.
		ombh2 = omegab*h**2
		omch2 = omegac*h**2
		
		input_parameters = np.array([ngal, siglogM, M0_over_M1, M1_over_Mmin, alpha,
									 A_conc, R_rescale,
								 	 sigma_8, H0, ombh2, omch2, w0, ns])

		prediction = emulate_fun(input_parameters)

		this_analytic_wp = compute_analytic_wp(omega_m, omegab, H0, ns,
											   w0, sigma_8,
											   redshift,
											   ngal, siglogM,
											   M1_over_Mmin, M0_over_M1, alpha, f_cen,
											   wp_binmin=rp_binmin,
											   wp_binmax=rp_binmax)
		prediction *= this_analytic_wp
		return prediction


	fiducial_prediction = emulate(params_fiducial)

	
	## make plots

	levels = 5
	
	for i, (param_name, pmin, pmax) in enumerate(zip(param_names, params_min, params_max)):
		
		print(f"{param_name} [{pmin}, {pmax}]")
		
		plt.figure()
	
		fidvalue = params_fiducial[i]
		if i==0:
			fidvalue *= 1e4

		plt.plot(rp_midbin, np.ones_like(fiducial_prediction), '--',
					label="fiducial value " + r"$=$" + f" {fidvalue:.3f}", color='blue')

		for j, pvalue in enumerate(np.linspace(pmin, pmax, levels, endpoint=True)):
		
			this_params = params_fiducial.copy()
			this_params[i] = pvalue
			prediction = emulate(this_params)
			
			if i==0:
				pvalue *= 1e4
			plt.plot(rp_midbin, prediction/fiducial_prediction,
						label=f"{pvalue:.3f}", color='black', alpha=float((j+1)/levels))
			
		plt.legend(loc='best')
		plt.xlabel(r'$r_p$ (Mpc/h)')
		plt.ylabel(r'$w_p / w_{p,fiducial}$')
		plt.title(f"{latex_labels[i]}")
		plt.xscale('log')
		plt.tight_layout()
		plt.savefig( Path(args.output_plot).with_suffix(f".{param_name}.pdf") )
		
		