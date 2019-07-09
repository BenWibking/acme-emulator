import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from preliminize import preliminize


if __name__ == '__main__':

	from pymultinest.analyse import Analyzer

	parser = argparse.ArgumentParser()
	parser.add_argument('--mcmc-ppd', help='emcee PPD output file')
	parser.add_argument('--multinest-dir', help='multinest output directory')
	parser.add_argument('ppd_plot_wp_file')
	parser.add_argument('ppd_plot_ds_file')
	parser.add_argument('inv_cov_file')
	parser.add_argument('cov_file')
	parser.add_argument('--discrepancy-plot', required=True)
	parser.add_argument('--watermark', default=None)
	parser.add_argument('--ppd-samples', default=20)

	args = parser.parse_args()
	PPD_nsamples = args.ppd_samples

	if args.multinest_dir is not None:

		n_dims = 14
		a = Analyzer(n_dims, outputfiles_basename=args.multinest_dir)

		bestfit_params = a.get_best_fit()['parameters']
		bestfit_lnL = a.get_best_fit()['log_likelihood']
		ppd_ml = np.array( bestfit_params[n_dims:] )

		ml_chisq = ppd_ml[-1]
		ml_wp, ml_ds = np.split(np.array(ppd_ml[:-1]), 2)

		print(f"best-fit log-likelihood = {bestfit_lnL}")
		print(f"best-fit chi-sq = {ml_chisq}")
		print(f"best-fit wp = {ml_wp}")
		print(f"best-fit DS = {ml_ds}")
		print(f"best-fit parameters = {bestfit_params[:n_dims]}")

		multinest_equal_samples_all = a.get_equal_weighted_posterior()
		multinest_equal_samples = multinest_equal_samples_all[:, :n_dims]
		multinest_ppd = multinest_equal_samples_all[:, n_dims:-1]

		## compute PPD statistics

		chisqs = multinest_ppd[:, -1]
		wps, deltasigmas = np.split(multinest_ppd[:, :-1], 2, axis=1)


	## read posterior predictive distributions

	if args.mcmc_ppd is not None:

		import h5py
		f = h5py.File(args.mcmc_ppd, 'r', swmr=True)
		wps = f['wp'][burn_in_samples:]
		deltasigmas = f['deltasigma'][burn_in_samples:]
		f.close()


	## read mock wp

	## TODO: make these input parameters
	rmin_mock, rmax_mock, _, wp_mock = np.loadtxt('../../lowz_mocks/data/lowz_corr_blinded.wp.txt', unpack=True)
	rmin_mock, rmax_mock, _, ds_mock = np.loadtxt('../../lowz_mocks/data/lowz_corr_blinded.ds.txt', unpack=True)

	r_mock = 0.5*(rmin_mock + rmax_mock)

	## TODO: make this a parameter
	rmin_cut = 0.6	# user-selected when fitting

	scale_mask = rmin_mock > rmin_cut
	mock_datavector = np.concatenate([wp_mock[scale_mask], ds_mock[scale_mask]])


	## read covariance matrix

	Cinv = np.loadtxt(args.inv_cov_file)	# inverse covariance matrix
	cov = np.loadtxt(args.cov_file)			# covariance matrix

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

	avg_wp = np.average(wps, axis=0, weights=importance_weights)
	std_wp = np.sqrt( np.average((wps - avg_wp)**2, axis=0, weights=importance_weights) )
	wp_samples = wps[rand_idx, :]

	avg_ds = np.average(deltasigmas_pointmass, axis=0, weights=importance_weights)
	avg_ds_nopointmass = np.average(deltasigmas, axis=0)
	std_ds = np.sqrt( np.average((deltasigmas_pointmass - avg_ds)**2, axis=0, weights=importance_weights) )
	ds_samples = deltasigmas_pointmass[rand_idx, :]



	def discrepancy_min(obs, ppd, skip_index=None):

		"""compute discrepancy distance function over PPD samples."""

		discrepancy_dist_i = np.zeros(ppd.shape[0])

		for j in range(ppd.shape[0]):
			yvec = (obs - ppd[j,:])
			discrepancy_dist_i[j] = np.dot(yvec.T, np.dot(Cinv, yvec))

		if skip_index is not None:
			discrepancy_dist_i[skip_index] = np.inf

		dist_i = np.min(discrepancy_dist_i)
		return dist_i


	def discrepancy_mean(obs, ppd, skip_index=None):

		"""compute discrepancy distance function over PPD samples."""

		discrepancy_dist_i = np.zeros(ppd.shape[0])

		for j in range(ppd.shape[0]):
			yvec = (obs - ppd[j,:])
			discrepancy_dist_i[j] = np.dot(yvec.T, np.dot(Cinv, yvec))

		dist_i = np.sum(discrepancy_dist_i)

		if skip_index is not None:
			dist_i /= (ppd.shape[0] - 1.0)
		else:
			dist_i /= ppd.shape[0]

		return dist_i


	def discrepancy_distance_distribution(this_datavector, this_ppd, function=None):

		N_ppd = this_ppd.shape[0]
		discrepancy_dist = np.zeros(N_ppd)

		for i in range(N_ppd):
			mean = this_ppd[i,:]
			mock_observed_ppd_i = np.random.multivariate_normal(mean, cov)
			discrepancy_dist[i] = function(mock_observed_ppd_i, this_ppd, skip_index=i)

		data_discrepancy = function(this_datavector, this_ppd)
		pvalue = np.count_nonzero(discrepancy_dist > data_discrepancy) / N_ppd

		return discrepancy_dist, data_discrepancy, pvalue


	discrepancy_dist_min, data_discrepancy_min, pvalue_min = discrepancy_distance_distribution(mock_datavector, ppd_restricted, function=discrepancy_min)

	discrepancy_dist_mean, data_discrepancy_mean, pvalue_mean = discrepancy_distance_distribution(mock_datavector, ppd_restricted, function=discrepancy_mean)

	print(f"")
	print(f"'discrepancy min distance' statistic [c.f. Gelman et al. 1996]")
	print(f"\tof mock data: {data_discrepancy_min}")
	print(f"\tMonte Carlo p-value: {pvalue_min:.5f} ({N_ppd} samples)")
	print(f"\tposterior mean: {np.mean(discrepancy_dist_min)}")
	print(f"\tposterior standard deviation: {np.sqrt(np.var(discrepancy_dist_min))}")
	print(f"")

	print(f"")
	print(f"'discrepancy mean distance' statistic [c.f. Gelman et al. 1996]")
	print(f"\tof mock data: {data_discrepancy_mean}")
	print(f"\tMonte Carlo p-value: {pvalue_mean:.5f} ({N_ppd} samples)")
	print(f"\tposterior mean: {np.mean(discrepancy_dist_mean)}")
	print(f"\tposterior standard deviation: {np.sqrt(np.var(discrepancy_dist_mean))}")
	print(f"")


	## plot posterior discrepancy distribution

	plt.figure()
	plt.hist((discrepancy_dist_min), density=True, bins=20)
	plt.axvline((data_discrepancy_min), linestyle='--', color='black',
				label=r"data $\textrm{min} \, \Delta \chi^2$" + \
									  f"\n({100*(1-pvalue_min):.1f} percentile)")

	plt.xlabel(r'$\textrm{min} \, \Delta \chi^2$ distance')
	plt.ylabel('probability density')
	legend = plt.legend(loc='best', frameon=True, shadow=True)
	legend.get_frame().set_facecolor('white')

	if args.watermark is not None:
		preliminize(text=args.watermark)
	plt.tight_layout()
	plt.savefig(str(Path(args.discrepancy_plot).with_suffix('.discrepancy_min.pdf')))
	plt.close()


	plt.figure()
	plt.hist((discrepancy_dist_mean), density=True, bins=20)
	plt.axvline((data_discrepancy_mean), linestyle='--', color='black',
				label=r"data $\langle \Delta \chi^2 \rangle$" + \
					  f"\n({100*(1-pvalue_mean):.1f} percentile)")

	plt.xlabel(r'$\langle \Delta \chi^2 \rangle$ distance')
	plt.ylabel('probability density')
	legend = plt.legend(loc='best', frameon=True, shadow=True)
	legend.get_frame().set_facecolor('white')

	if args.watermark is not None:
		preliminize(text=args.watermark)
	plt.tight_layout()
	plt.savefig(str(Path(args.discrepancy_plot).with_suffix('.discrepancy_mean.pdf')))
	plt.close()


	## plot PPD for wp

	plt.figure()
	plt.xscale('log')
	plt.xlabel(r'$r_p$ [Mpc/h]')
	plt.ylabel(r'$r \times w_p$ [Mpc/h]')

	if args.mcmc_ppd is not None or args.multinest_dir is not None:
		for i in range(wp_samples.shape[0]):
			plt.plot(r_mock, r_mock*wp_samples[i,:], color='gray', alpha=0.5, linewidth=0.5)

		plt.errorbar(r_mock, r_mock*avg_wp, yerr=r_mock*std_wp,
					 color='black', label='posterior mean')

	plt.errorbar(r_mock[scale_mask], r_mock[scale_mask]*wp_mock[scale_mask],
				 yerr=r_mock[scale_mask]*wp_errbar,
				 fmt='.-', color='red', label='data')

	plt.axvline(x=rmin_cut, linestyle='--', color='black')
	plt.axvspan(0, rmin_cut, alpha=0.5, color='orange')

	plt.ylim( 0.9*np.min(r_mock*wp_mock), 1.1*np.max(r_mock*wp_mock) )

	plt.legend(loc='lower right')

	if args.watermark is not None:
		preliminize(text=args.watermark)
	plt.tight_layout()
	plt.savefig(args.ppd_plot_wp_file)
	plt.close()


	## plot PPD for DS

	plt.figure()
	plt.xscale('log')
	plt.xlabel(r'$r_p$ [Mpc/h]')
	plt.ylabel(r'$r \times \Delta\Sigma$')

	if args.mcmc_ppd is not None or args.multinest_dir is not None:
		for i in range(ds_samples.shape[0]):
			plt.plot(r_mock, r_mock*ds_samples[i,:], color='gray', alpha=0.5, linewidth=0.5)

		plt.errorbar(r_mock, r_mock*avg_ds, yerr=r_mock*std_ds, color='black',
					 label='posterior mean')
		plt.plot(r_mock, r_mock*avg_ds_nopointmass, '--', color='black',
					 label='posterior mean\n(no point mass)')

	plt.errorbar(r_mock[scale_mask], r_mock[scale_mask]*ds_mock[scale_mask],
				 yerr=r_mock[scale_mask]*ds_errbar,
				 fmt='.-', color='red', label='data')

	plt.axvline(x=rmin_cut, linestyle='--', color='black')
	plt.axvspan(0, rmin_cut, alpha=0.5, color='orange')

	plt.ylim( 0.8*np.min(r_mock*ds_mock), 1.2*np.max(r_mock*ds_mock) )

	plt.legend(loc='upper right')

	if args.watermark is not None:
		preliminize(text=args.watermark)
	plt.tight_layout()
	plt.savefig(args.ppd_plot_ds_file)
	plt.close()


	## plot lensing is low for DS

	plt.figure()
	plt.xscale('log')
	plt.xlabel(r'$r_p$ [Mpc/h]')
	plt.ylabel(r'$\Delta\Sigma$ ratio')

	if args.mcmc_ppd is not None or args.multinest_dir is not None:
		for i in range(ds_samples.shape[0]):
			plt.plot(r_mock, ds_samples[i,:]/ds_mock, color='gray', alpha=0.5, linewidth=0.5)

		plt.errorbar(r_mock, avg_ds/ds_mock, yerr=std_ds/ds_mock, color='black',
					 label='posterior mean')

	plt.plot(r_mock, ml_ds/ds_mock, label='maximum likelihood')
	plt.plot(r_mock, np.ones_like(r_mock), '.-', color='red', label='data')

	plt.axvline(x=rmin_cut, linestyle='--', color='black')
	plt.axvspan(0, rmin_cut, alpha=0.5, color='orange')

	plt.ylim(0.5, 1.5)

	plt.legend(loc='best')

	if args.watermark is not None:
		preliminize(text=args.watermark)
	plt.tight_layout()
	plt.savefig(str(Path(args.ppd_plot_ds_file).with_suffix('.ds_ratio.pdf')))
	plt.close()

