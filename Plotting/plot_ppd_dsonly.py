import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from pymultinest.analyse import Analyzer
from preliminize import preliminize

matplotlib.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}']

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--multinest-dir', nargs='*', help='multinest output directories')
	parser.add_argument('--inv_cov', nargs='*')
	parser.add_argument('--cov', nargs='*')
	parser.add_argument('--label', nargs='*')
	parser.add_argument('ppd_plot_ds_file')

	parser.add_argument('--watermark', default=None)
	parser.add_argument('--ppd-samples', default=20)

	args = parser.parse_args()
	PPD_nsamples = args.ppd_samples


	## read mock wp

	rmin_cut = 0.6	# user-selected when fitting

	rmin_mock, rmax_mock, _, wp_mock = np.loadtxt('../../lowz_mocks/data/lowz_corr_blinded.wp.txt', unpack=True)
	rmin_mock, rmax_mock, _, ds_mock = np.loadtxt('../../lowz_mocks/data/lowz_corr_blinded.ds.txt', unpack=True)

	r_mock = 0.5*(rmin_mock + rmax_mock)
	scale_mask = rmin_mock > rmin_cut
	mock_datavector = np.concatenate([wp_mock[scale_mask], ds_mock[scale_mask]])


	## make plot

	plt.figure()
	plt.xscale('log')
	plt.xlabel(r'$r_p$ [Mpc/h]')
	plt.ylabel(r'$r \times \Delta\Sigma$')

	color_cycle = ['black', 'blue', 'green', 'red', 'brown']
	style_cycle = ['--', '-.', ':', '-..']

	for i, (multinest_dir, inv_cov_file, cov_file, label) in enumerate(zip(args.multinest_dir, args.inv_cov, args.cov, args.label)):

		print(f"reading {multinest_dir}")

		line_color = color_cycle[np.mod(i, len(color_cycle))]
		line_style = style_cycle[np.mod(i, len(style_cycle))]

		## read samples
	
		n_dims = 14
		a = Analyzer(n_dims, outputfiles_basename=multinest_dir)
	
		bestfit_params = a.get_best_fit()['parameters']
		bestfit_lnL = a.get_best_fit()['log_likelihood']
		ppd_ml = np.array( bestfit_params[n_dims:] )
	
		ml_chisq = ppd_ml[-1]
		ml_wp, ml_ds = np.split(np.array(ppd_ml[:-1]), 2)
	
		multinest_equal_samples_all = a.get_equal_weighted_posterior()
		multinest_equal_samples = multinest_equal_samples_all[:, :n_dims]
		multinest_ppd = multinest_equal_samples_all[:, n_dims:-1]
	
	
		## compute PPD statistics
	
		chisqs = multinest_ppd[:, -1]
		wps, deltasigmas = np.split(multinest_ppd[:, :-1], 2, axis=1)


		## read covariance matrix
	
		Cinv = np.loadtxt(inv_cov_file)	# inverse covariance matrix
		cov = np.loadtxt(cov_file)		# covariance matrix
	
		errbar = np.sqrt(np.diag(cov))
		wp_errbar, ds_errbar = np.split(errbar, 2)


		## compute Cinv_{point_mass} via the Sherman-Morrison formula
	
		point_mass_term = (1.0 / r_mock**2)
	
		Cinv_full = np.linalg.inv(cov)
		Cinv_pointmass = Cinv_full - Cinv
	

		## compute posterior predictive distribution (PPD)
	
		N_ppd = wps.shape[0]
	
		point_masses = np.random.normal(loc=0.0, scale=20.0, size=N_ppd)[:, np.newaxis] \
						* point_mass_term[np.newaxis, :]
		deltasigmas_pointmass = deltasigmas + point_masses
	
		ppd_restricted = np.block([wps[:, scale_mask], deltasigmas[:, scale_mask]])
		ppd_pointmass = np.block([wps[:, scale_mask], deltasigmas_pointmass[:, scale_mask]])
	
		importance_weights = np.zeros(N_ppd)
	
		for j in range(N_ppd):
			yvec_pm = mock_datavector - ppd_pointmass[j,:]
			chisq_pointmass = np.dot(yvec_pm.T, np.dot(Cinv_pointmass, yvec_pm))
			importance_weights[j] = np.exp( -0.5*(chisq_pointmass) )
	
		sum_weights = np.sum(importance_weights)
		importance_weights /= sum_weights
	
	
		## compute PPD statistics
	
		rand_idx = np.random.choice(range(N_ppd), size=PPD_nsamples, p=importance_weights)
	
		avg_ds = np.average(deltasigmas_pointmass, axis=0, weights=importance_weights)
		std_ds = np.sqrt( np.average((deltasigmas_pointmass - avg_ds)**2, axis=0,
										weights=importance_weights) )
		ds_samples = deltasigmas_pointmass[rand_idx, :]
	
	
		## plot PPD for DS
	
		plt.errorbar(r_mock, r_mock*avg_ds, yerr=r_mock*std_ds, color=line_color,
					 linestyle=line_style, label=f"{label}")
	

#	plt.plot(r_mock, r_mock*ds_mock, '.-', color='red', label='data')

	plt.errorbar(r_mock[scale_mask], r_mock[scale_mask]*ds_mock[scale_mask],
				 yerr=r_mock[scale_mask]*ds_errbar,
				 fmt='.-', color='red', label='data')


	## output plot

	plt.axvline(x=rmin_cut, linestyle='--', color='black')
	plt.axvspan(0, rmin_cut, alpha=0.5, color='orange')

	plt.legend(loc='upper left')
	plt.ylim( 5.5, 12.0 )

	if args.watermark is not None:
		preliminize(text=args.watermark)
	plt.tight_layout()
	plt.savefig(args.ppd_plot_ds_file)
	plt.close()




