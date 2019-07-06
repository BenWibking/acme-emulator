import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from pymultinest.analyse import Analyzer
from predict_emulator import get_emulate_fun
from preliminize import preliminize


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('ppd_plot_hod_file')
	parser.add_argument('--multinest-dir', help='multinest output directory')
	parser.add_argument('--emulator', required=True)
	parser.add_argument('--ppd-samples', default=20)
	parser.add_argument('--watermark')

	args = parser.parse_args()


	## read posterior samples

	PPD_nsamples = args.ppd_samples
	n_dims = 14
	a = Analyzer(n_dims, outputfiles_basename=args.multinest_dir)
	multinest_equal_samples_all = a.get_equal_weighted_posterior()
	multinest_equal_samples = multinest_equal_samples_all[:, :n_dims]


	## compute PPD statistics

	posterior_parameters = multinest_equal_samples

	emufun, _, _ = get_emulate_fun(args.emulator, load_scalar=True)
	logmass = np.linspace(12.5, 15.3, 256)
	M = 10**logmass

	ppdmax = posterior_parameters.shape[0]
	logMmin = np.zeros( ppdmax )
	ppd_hod = np.zeros( (ppdmax, logmass.shape[0]) )

	def tabulated_hod(logMmin, siglogM, M1_over_Mmin, M0_over_M1, alpha):
		Mmin = 10**logMmin
		M1 = M1_over_Mmin * Mmin
		M0 = M0_over_M1 * M1
		mean_cen = 0.5 * ( 1.0 + scipy.special.erf((logmass - logMmin)/siglogM) )
		mean_sat = np.zeros_like(mean_cen)
		mean_sat[M > M0] = mean_cen[M > M0] * (((M[M > M0] - M0) / M1)**alpha)
		return (mean_cen + mean_sat)
	
	for i in range(ppdmax):

		if i % 100 == 0:
			print(i)

		## extract parameters from posterior_parameters
		params = posterior_parameters[i,:]

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
		ds_calibration	 = params[13]

		omegac = omega_m - omegab
		h = H0/100.
		ombh2 = omegab*h**2
		omch2 = omegac*h**2		

		input_parameters = np.array([ngal, siglogM, M0_over_M1, M1_over_Mmin, alpha,
									 A_conc, R_rescale,
								 	 sigma_8, H0, ombh2, omch2, w0, ns])

		logMmin[i] = emufun(input_parameters)
		ppd_hod[i,:] = tabulated_hod(logMmin[i], siglogM, M1_over_Mmin, M0_over_M1, alpha)


	mean_ppd_hod = np.average(ppd_hod, axis=0)
	std_ppd_hod = np.sqrt( np.average((ppd_hod - mean_ppd_hod)**2, axis=0) )
	samples = np.random.randint(ppdmax, size=PPD_nsamples)

	
	## plot PPD for wp

	plt.figure()
	plt.yscale('log')
	plt.ylim( (0.1, 15.0) )
	plt.xlabel(r'log halo mass [$M_{\odot}$]')
	plt.ylabel(r'mean halo occupation')

	for i in range(len(samples)):
		idx = samples[i]
		plt.plot(logmass, ppd_hod[idx, :], color='gray', linewidth=0.5, zorder=-100)

	plt.plot(logmass, mean_ppd_hod, color='black', label='posterior mean')
	plt.fill_between(logmass, mean_ppd_hod-std_ppd_hod, mean_ppd_hod+std_ppd_hod,
					 alpha=0.5, zorder=-99, color='orange')
	plt.plot(logmass, np.ones_like(logmass), '--', color='black')

#	plt.title('posterior predictive distribution')
	plt.legend(loc='lower right')

	if args.watermark is not None:
		preliminize(text=args.watermark)

	plt.tight_layout()
	plt.savefig(args.ppd_plot_hod_file)
	plt.close()

	## plot fractional uncertainty in distribution

	plt.figure()
	plt.ylim( (-1, 1) )
	plt.xlabel(r'log halo mass [$M_{\odot}$]')
	plt.ylabel(r'fractional uncertainty')

	plt.fill_between(logmass, -std_ppd_hod/mean_ppd_hod, std_ppd_hod/mean_ppd_hod,
					 alpha=0.5, zorder=-99, color='orange')
	plt.plot(logmass, np.zeros_like(logmass), color='black', label='posterior mean')

#	plt.title('posterior predictive distribution')
	plt.legend(loc='lower right')

	if args.watermark is not None:
		preliminize(text=args.watermark)

	plt.tight_layout()
	plt.savefig(Path(args.ppd_plot_hod_file).with_suffix('.uncertainty.pdf'))
	plt.close()