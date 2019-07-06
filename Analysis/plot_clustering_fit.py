import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.special
import scipy.interpolate
import math
import h5py
import matplotlib.pyplot

from compute_hod import camb_linear_pk, cM_Correa2015, compute_xi_1halo, compute_xi_2halo, \
		compute_linear_bias, dndm_tinker, compute_xigm_1halo, compute_xigm_2halo

from create_mocks import wp, DeltaSigma


if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('hod_tab_file')
	parser.add_argument('fit_wp_file')
	parser.add_argument('predicted_DS_file')
	parser.add_argument('fit_hod_file')
	parser.add_argument('param_file')
	args = parser.parse_args()
	
	## compute power spectrum for fiducial cosmology

	omegam_bolshoi = 0.27
	redshift_bolshoi = 0.1
	sigma8_bolshoi = 0.82
	H0_bolshoi = 70.
	ns_bolshoi = 0.95
	omegab_bolshoi = 0.0469

	h_bolshoi = H0_bolshoi/100.
	omegac_bolshoi = omegam_bolshoi-omegab_bolshoi
	w0_bolshoi = -1.0

	omega_m = omegam_bolshoi
	redshift = redshift_bolshoi
	sigma_8 = sigma8_bolshoi
	ns = ns_bolshoi
	ombh2 = omegab_bolshoi*h_bolshoi**2
	omch2 = omegac_bolshoi*h_bolshoi**2
	w0 = w0_bolshoi
	H0 = H0_bolshoi
	h = H0/100.

	k, P = camb_linear_pk(ombh2=ombh2, omch2=omch2, H0=H0, ns=ns, w0=w0,
						  sigma8=sigma_8, redshift=redshift)

	
	## read observed wp
	
	print("reading in observed wp...", end='', flush=True)
	
	mock_wp = np.loadtxt()

	
	## read covariance matrix

#	cov = np.loadtxt('./data/wpgg_cov_sim_smoothed.txt')


	## plot covariance matrix

	import matplotlib.pyplot as plt
	plt.figure()
	im = plt.pcolormesh(corr)
	plt.savefig('./output/wp_cov.pdf')
	plt.close()

	plt.figure()
	frac_error = np.sqrt(np.diag(cov))/mock_wp
	plt.fill_between(wp_bins, -frac_error, frac_error, alpha=0.2, label='error bar',
					 facecolor='black')
	plt.xscale('log')
	plt.xlabel(r'$r_p$ (Mpc/$h$)')
	plt.ylabel(r'fractional error')
	plt.legend(loc='best')
	plt.savefig('./output/wp_cov_diagonal.pdf')
	plt.close()


	## params: logMmin, logM1, siglogM, alpha
    
	print("using pre-computed maximum likelihood...")

	## read parameters file from previous run
		
	import configparser
	config = configparser.ConfigParser()
	config.read(args.param_file)
	params = config['params']
		
	logMmin = float(params['logMmin'])
	logM1 = float(params['logM1'])
	siglogM = float(params['siglogM'])
	alpha = float(params['alpha'])
	maximum_likelihood_params = [logMmin, logM1, siglogM, alpha]
	
	
	params_ml = maximum_likelihood_params
	Ncen_ml, Nsat_ml, Ntot_ml = hod_from_parameters(params_ml[0], 0.,
													params_ml[1], params_ml[2], params_ml[3])
	fit_wp, fit_r, fit_xigg, fit_ngal, fit_xigg_cs, fit_xigg_ss, fit_wp_2halo, fit_wp_cs, fit_wp_ss = \
													analytic_wp(Ncen_ml, Nsat_ml, Ntot_ml,
													binmin=binmin, binmax=binmax,
													k=k, P=P, Omega_M=omega_m, redshift=redshift)
	

	## three-panel plot: R * w_p, HOD, and R * DeltaSigma
	
	f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4))
	
	## R * wp
	
	ax1.set_xscale('log')
	ax1.set_xlabel(r'$r_p$ (Mpc/$h$)')
	ax1.set_ylabel(r'$r_p \times w_p$ (Mpc/$h$)')
	r = 0.5*(binmin + binmax)
	
	ax1.plot(r, r * fit_wp, '--',color='black')
	ax1.errorbar(r, r * avg_wp, yerr=r*std_wp, fmt='--', color='grey',
				 label=r'posterior mean')
	ax1.plot(r, r * mock_wp,  color='black')
	
	ax1.plot(r, r * fit_wp_cs, '--', color='red', label=r'predicted 1-halo c-s')
	ax1.plot(r, r * fit_wp_ss, '--', color='blue', label=r'predicted 1-halo s-s')
	ax1.plot(r, r * fit_wp_2halo, '--', label=r'predicted 2-halo', color='green')
	
	ax1.plot(r, r * wp_cs_mock, color='red', alpha=0.5)
	ax1.plot(r, r * wp_ss_mock, color='blue', alpha=0.5)
	ax1.plot(r, r * wp_2halo_mock, label=r'mock 2-halo', color='green', alpha=0.5)
	
	ax1.legend(loc='upper left')
	
	## HOD
	
	ax2.set_xscale('log')
	ax2.set_yscale('log')
	ax2.set_xlim(1e12, 1e15)
	ax2.set_ylim(0.01, 10)
	ax2.set_xlabel(r'halo mass $M_{\odot}$')
	ax2.set_ylabel(r'halo occupation')
	
	ax2.plot(massbins, central_hod, '--', label='predicted central', color='red', alpha=0.5)
	ax2.plot(massbins, central_hod*sat_hod, '--', label='predicted satellite', color='blue', alpha=0.5)
	ax2.plot(massbins, central_hod + central_hod*sat_hod, '--',
			 label='predicted total', color='black', alpha=0.5)
			 
	ax2.plot(masstab_mock, central_hod_mock, label='mock central', color='red')
	ax2.plot(masstab_mock, central_hod_mock*satellite_hod_mock, label='mock satellite', color='blue')
	ax2.plot(masstab_mock, central_hod_mock + central_hod_mock*satellite_hod_mock,
			 label='mock total', color='black')
			 
	ax2.legend(loc='best')
	
	## R * DeltaSigma
	
	ax3.set_xscale('log')
	ax3.set_xlabel(r'$r_p$ (Mpc/$h$)')
	ax3.set_ylabel(r'$r_p \times \Delta \Sigma$')
	
	ax3.plot(r, r * predicted_DS, '--', color='black')
	ax3.errorbar(r, r * avg_ds, yerr=r*std_ds, fmt='--', label=r'posterior mean', color='grey')	
	ax3.plot(r, r * mock_DS, color='black')
	
	ax3.plot(r, r * predicted_DS_cen, '--', color='red')
	ax3.plot(r, r * predicted_DS_sat, '--', color='blue')
	ax3.plot(r, r * predicted_DS_2halo, '--', label=r'predicted 2-halo', color='green')
	ax3.plot(r, r * mock_DS_cen, color='red', alpha=0.5)
	ax3.plot(r, r * mock_DS_sat, color='blue', alpha=0.5)
	ax3.plot(r, r * mock_DS_2halo, label=r'mock 2-halo', color='green', alpha=0.5)
	ax3.legend(loc='upper right')
	
	f.suptitle(r'Zheng05 fit to $w_p$ from iHOD mock ($\alpha_{{g-r}} = {}$); $\chi^2 / \textrm{{dof}} = {:.3f}$; fit $n_g = {:.6f}$; mock $n_g = {:.6f}$'.format(args.input_alpha, chi_sq_over_dof, fit_ngal, ngal_mock))
	f.savefig(args.panel_plot_file)


	## plot lensing-is-low
	
	cov_DS = np.diag((0.10 * mock_DS)**2)   # assume 10% diagonal error bars (shape noise dominated)

	plt.figure()
	plt.xscale('log')
	plt.xlabel(r'$r_p$ (Mpc/$h$)')
	plt.ylim(0.8, 1.3)
	plt.xlim(0.3, 30.0)
	plt.ylabel(r'$\Delta\Sigma_{\textrm{predict}} / \Delta\Sigma_{\textrm{mock}}$')
	plt.plot(r, np.ones(r.shape) * 1.0, color='black')

	plt.plot(r, predicted_DS / mock_DS, '--', label='maximum-likelihood',
			 color='red')
	
	plt.legend(loc='best')
	plt.title(r'"lensing is low comparison" ($\alpha = {}$)'.format(args.input_alpha))
	plt.tight_layout()
	plt.savefig('./output/lensing_is_low.pdf')

