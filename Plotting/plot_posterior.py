import numpy as np
import matplotlib.pyplot as plt
import getdist
import getdist.plots
import argparse
from pylatexenc.latexencode import utf8tolatex
from pathlib import Path
from preliminize import preliminize


## BigMDPL values (used for LOWZ mock)

true_S_8 = 1.0
true_sigma8 = 0.8228
true_omegam = 0.3107115
true_H0 = 67.77
true_omegab = 0.048206
true_ns = 0.96
true_w0 = -1.0
true_Alensing = 1.0


def get_samples(posterior_samples_weights, run_label=None, set_labels=True):

	names = ["ngal", "M0/M1", "M1/Mmin", "siglogM", "alpha", "Aconc", "Rrescale",
			   "S8-Singh", "sigma8", "omegam", "H0", "ns", "omegab", "w0", "Alensing"]	
	cosmo_names = ["S8-Singh", "sigma8", "omegam", "H0", "ns", "omegab", "w0", "Alensing"]
	cosmo_values = [true_S_8, true_sigma8, true_omegam, true_H0, true_ns, true_omegab, true_w0, true_Alensing]
	cosmo_pairs = list(zip(cosmo_names, cosmo_values))
	hod_names = ["ngal", "M0/M1", "M1/Mmin", "siglogM", "alpha", "Aconc", "Rrescale"]
		
	labels=[r"n_{g} \times 10^{4}",
				r"M_0 / M_1",
				r"M_1 / M_{\text{min}}",
				r"\sigma_{\log M}",
				r"\alpha",
				r"A_{\text{conc}}",
				r"R_{\text{rescale}}",
				r"S_8",
				r"\sigma_8",
				r"\Omega_m",
				r"H_0",
				r"n_s",
				r"\Omega_b",
				r"w_0",
				r"A_{\text{lensing}}"]
				
	assert len(labels) == len(names)
	latex_labels = [r"$" + label + r"$" for label in labels]

	latex_labels_dict = dict(zip(names, latex_labels))

	mysamples = []
	nonskip_params = []

	for samples, weights, label in posterior_samples_weights:

		samples[:, 0] *= 1e4	# rescale n_gal so it is readable
	
		# add derived S_8 parameter [S_8 == (\Omega_m/0.3)^{0.6} * \sigma_8]
	
		Omega_m_samples = samples[:, -7]
		sigma_8_samples = samples[:, -6]
		S_8_samples = (Omega_m_samples / true_omegam)**(0.6) * \
						(sigma_8_samples / true_sigma8)
	
		new_samples = np.column_stack((samples[:, :-7],
									   S_8_samples, sigma_8_samples, Omega_m_samples,
									   samples[:, -5:]))
	
		# for MultiNest samples, this works fine -- do not use for emcee samples!
		prior_min = np.min(new_samples, axis=0)
		prior_max = np.max(new_samples, axis=0)
	
		assert len(prior_min) == len(prior_max)
		assert len(prior_min) == len(names)
	
		prior_range = dict([(names[i], [prior_min[i], prior_max[i]]) \
							 for i in range(len(prior_min))])
		
		for this_param in prior_range:
			this_range = prior_range[this_param][1] - prior_range[this_param][0]
			if this_range > 0.0:
				nonskip_params.append(this_param)

		if set_labels == False:
			labels = None

		mysamples.append( getdist.MCSamples(samples=new_samples, weights=weights,
									  		ranges=prior_range,
									  		names=names, labels=labels,
									  		label=label) )

	nonskip_params = list(set(nonskip_params))
	myhod_names = [name for name in hod_names if name in nonskip_params]
	mycosmo_names = [name for (name,val) in cosmo_pairs if name in nonskip_params]
	mycosmo_values = [val for (name,val) in cosmo_pairs if name in nonskip_params]
		
	return mysamples, myhod_names, mycosmo_names, mycosmo_values, latex_labels_dict


def plot_posteriors(posterior_samples_weights, 
					filename=None, watermark=None, plot_truth=True):

	posteriors_mysamples, hod_names_plot, cosmo_names_plot, cosmo_values_plot, latex_labels_dict = get_samples(posterior_samples_weights)
	
	
	## make the plot

	import matplotlib as mpl
	mpl.rcParams['text.usetex'] = True
	mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] # for \text command

	def plot_true_vertical(subplot_list, true_value):
			for ax in subplot_list:
					if ax is not None:
						ax.axvline(true_value, color='black', ls='--', lw=0.75)

	def plot_true_horizontal(subplot_list, true_value):
			for ax in subplot_list:
					if ax is not None:
						ax.axhline(true_value, color='black', ls='--', lw=0.75)


	## make hod-only plot

	g = getdist.plots.getSubplotPlotter()
	g.triangle_plot(posteriors_mysamples, filled=True, params=hod_names_plot)

	if len(posteriors_mysamples) == 1:
		for i, ax in enumerate(np.diag(g.subplots)):
			ax.set_title(f"${posteriors_mysamples[0].getInlineLatex(hod_names_plot[i])}$")

	if watermark is not None:
		preliminize(text=watermark)	
	g.export( str(Path(filename).with_suffix('.hod.pdf')) )


	## plot mixed posterior

	g = getdist.plots.getSubplotPlotter()
	g.rectangle_plot(cosmo_names_plot, hod_names_plot, roots=posteriors_mysamples, filled=True)

	if plot_truth == True:
		for i, param_true in enumerate(cosmo_values_plot):
			plot_true_vertical(g.subplots[:, i], param_true)

	if watermark is not None:
		preliminize(text=watermark)	
	g.export( str(Path(filename).with_suffix('.mixed_params.pdf')) )
	
	
	## read in Planck samples
	
	fiducial_samples = posteriors_mysamples[0]
	fiducial_stats = fiducial_samples.getMargeStats()
	fiducial_s8_mean = fiducial_stats.parWithName('S8-Singh').mean
	fiducial_s8_err  = fiducial_stats.parWithName('S8-Singh').err
	print(f"LOWZ S_8 = {fiducial_s8_mean} +/- {fiducial_s8_err}") 
	
	planck_base = "/home/payne/wibking.1/Planck_base_plikHM_TTTEEE_lowl-lowE/base/plikHM_TTTEEE_lowl_lowE/base_plikHM_TTTEEE_lowl_lowE"
	
	planck_samples = getdist.loadMCSamples(planck_base)
	planck_samples.name_tag = r"Planck 2018 ($\Lambda$CDM)"
	
	p = planck_samples.getParams()
	planck_samples.addDerived((p.sigma8 / true_sigma8)*(p.omegam / true_omegam)**0.6,
								'S8-Singh', label=r"$S_8$")
	planck_samples.addDerived(p.omegabh2 / (p.H0/100.)**2,
								'omegab', label=r"$\Omega_b$")

	planck_stats = planck_samples.getMargeStats()
	planck_s8_mean = planck_stats.parWithName('S8-Singh').mean
	planck_s8_err  = planck_stats.parWithName('S8-Singh').err
	print(f"Planck 2018 S_8 = {planck_s8_mean} +/- {planck_s8_err}")
	
	s8_tension = (planck_s8_mean - fiducial_s8_mean) / np.sqrt(planck_s8_err**2 + fiducial_s8_err**2)
	print(f"\t(Planck 2018 - LOWZ) S_8 = {planck_s8_mean - fiducial_s8_mean} +/- {np.sqrt(planck_s8_err**2 + fiducial_s8_err**2)}")
	print(f"\tTension: {s8_tension} sigma")
	print(f"")
	
	posteriors_mysamples.append(planck_samples)


	## read in DES Y1 samples
	
	des_names = ['omegam', 'H0', 'omegab', 'ns', 'w0', 'sigma8']
	des_raw_samples = np.genfromtxt("/home/payne/wibking.1/desy1_chains/d_w3.txt",
									comments="#",
									usecols=(0, 1, 2, 3, 6, -3),
									names=None)
	des_weights = np.genfromtxt("/home/payne/wibking.1/desy1_chains/d_w3.txt",
									comments="#",
									usecols=(-1),
									names=None)
	des_raw_samples[:, 1] *= 100.0
	des_samples = getdist.MCSamples(samples=des_raw_samples, weights=des_weights, loglikes=None,
									ranges=None, names=des_names,
									label=r'DES Y1 3x2pt (wCDM$+\nu$)')
	p = des_samples.getParams()
	des_samples.addDerived((p.sigma8 / true_sigma8)*(p.omegam / true_omegam)**0.6,
								'S8-Singh', label=r"$S_8$")
								
	des_stats = des_samples.getMargeStats()
	des_s8_mean = des_stats.parWithName('S8-Singh').mean
	des_s8_err  = des_stats.parWithName('S8-Singh').err
	print(f"DES Y1 wCDM S_8 = {des_s8_mean} +/- {des_s8_err}")
	
	s8_tension = (planck_s8_mean - des_s8_mean) / np.sqrt(planck_s8_err**2 + des_s8_err**2)
	print(f"\t(Planck 2018 - DES Y1) S_8 = {planck_s8_mean - des_s8_mean} +/- {np.sqrt(planck_s8_err**2 + des_s8_err**2)}")
	print(f"\tTension: {s8_tension} sigma")
	print(f"")

	s8_tension = (fiducial_s8_mean - des_s8_mean) / np.sqrt(fiducial_s8_err**2 + des_s8_err**2)
	print(f"\t(LOWZ - DES Y1) S_8 = {fiducial_s8_mean - des_s8_mean} +/- {np.sqrt(fiducial_s8_err**2 + des_s8_err**2)}")
	print(f"\tTension: {s8_tension} sigma")
	
	posteriors_mysamples.append(des_samples)
	
	
	## read in DES Y1 (LCDM, fixed nu) samples
	
	des_names = ['omegam', 'H0', 'omegab', 'ns', 'sigma8']
	des_raw_samples = np.genfromtxt("/home/payne/wibking.1/desy1_chains/2pt_NG_mcal_1110.fits_d_l3_fixednu_chain.txt",
									comments="#",
									usecols=(0, 1, 2, 3, -3),
									names=None)
	des_weights = np.genfromtxt("/home/payne/wibking.1/desy1_chains/2pt_NG_mcal_1110.fits_d_l3_fixednu_chain.txt",
									comments="#",
									usecols=(-1),
									names=None)
	des_raw_samples[:, 1] *= 100.0
	des_samples = getdist.MCSamples(samples=des_raw_samples, weights=des_weights, loglikes=None,
									ranges=None, names=des_names,
									label=r'DES Y1 3x2pt ($\Lambda$CDM, fixed $\nu$)')
	p = des_samples.getParams()
	des_samples.addDerived((p.sigma8 / true_sigma8)*(p.omegam / true_omegam)**0.6,
								'S8-Singh', label=r"$S_8$")
								
	des_stats = des_samples.getMargeStats()
	des_s8_mean = des_stats.parWithName('S8-Singh').mean
	des_s8_err  = des_stats.parWithName('S8-Singh').err
	print(f"DES Y1 wCDM S_8 = {des_s8_mean} +/- {des_s8_err}")
	
	s8_tension = (planck_s8_mean - des_s8_mean) / np.sqrt(planck_s8_err**2 + des_s8_err**2)
	print(f"\t(Planck 2018 - DES Y1 [fixed nu]) S_8 = {planck_s8_mean - des_s8_mean} +/- {np.sqrt(planck_s8_err**2 + des_s8_err**2)}")
	print(f"\tTension: {s8_tension} sigma")
	print(f"")

	s8_tension = (fiducial_s8_mean - des_s8_mean) / np.sqrt(fiducial_s8_err**2 + des_s8_err**2)
	print(f"\t(LOWZ - DES Y1 [fixed nu]) S_8 = {fiducial_s8_mean - des_s8_mean} +/- {np.sqrt(fiducial_s8_err**2 + des_s8_err**2)}")
	print(f"\tTension: {s8_tension} sigma")
	
	posteriors_mysamples.append(des_samples)


	## make cosmo-only plot

	g = getdist.plots.getSubplotPlotter()
	g.triangle_plot(posteriors_mysamples, filled=False, params=cosmo_names_plot,
					figure_legend_frame=True)

	if len(posteriors_mysamples) == 1:
		for i, ax in enumerate(np.diag(g.subplots)):
			ax.set_title(f"${posteriors_mysamples[0].getInlineLatex(cosmo_names_plot[i])}$")

	if plot_truth == True:
		for i, (param_name, param_true) in enumerate(zip(cosmo_names_plot, cosmo_values_plot)):
			print(f"[{i}] {param_name} = {param_true}")
			plot_true_vertical(g.subplots[:, i], param_true)
			plot_true_horizontal(g.subplots[i, :i], param_true)

	if watermark is not None:
		preliminize(text=watermark)	
	g.export( str(Path(filename).with_suffix('.cosmo.pdf')) )


	## make (S8, sigma8, omegaM, A_lensing)-only plot

	scosmo_names = ["S8-Singh", "sigma8", "omegam", "Alensing"]
#	scosmo_names = ["S8-Singh", "sigma8", "omegam"]
	scosmo_values = [val for (name,val) in zip(cosmo_names_plot,cosmo_values_plot) if name in scosmo_names]

	g = getdist.plots.getSubplotPlotter()
	g.triangle_plot(posteriors_mysamples, filled=False, params=scosmo_names,
					figure_legend_frame=True)

	if len(posteriors_mysamples) == 1:
		for i, ax in enumerate(np.diag(g.subplots)):
			ax.set_title(f"${posteriors_mysamples[0].getInlineLatex(scosmo_names[i])}$")

	if plot_truth == True:
		for i, (param_name, param_true) in enumerate(zip(scosmo_names, scosmo_values)):
			print(f"[{i}] {param_name} = {param_true}")
			plot_true_vertical(g.subplots[:, i], param_true)
			plot_true_horizontal(g.subplots[i, :i], param_true)

	if watermark is not None:
		preliminize(text=watermark)	
	g.export( str(Path(filename).with_suffix('.scosmo.pdf')) )


	## make (S8, omegaM)-only plot

	scosmo_names = ["S8-Singh", "omegam"]
	scosmo_values = [val for (name,val) in zip(cosmo_names_plot,cosmo_values_plot) if name in scosmo_names]

	g = getdist.plots.getSubplotPlotter()
	g.triangle_plot(posteriors_mysamples, filled=False, params=scosmo_names,
					figure_legend_frame=True, legend_labels=[])

	if len(posteriors_mysamples) == 1:
		for i, ax in enumerate(np.diag(g.subplots)):
			ax.set_title(f"${posteriors_mysamples[0].getInlineLatex(scosmo_names[i])}$")

	if plot_truth == True:
		for i, (param_name, param_true) in enumerate(zip(scosmo_names, scosmo_values)):
			print(f"[{i}] {param_name} = {param_true}")
			plot_true_vertical(g.subplots[:, i], param_true)
			plot_true_horizontal(g.subplots[i, :i], param_true)

	if watermark is not None:
		preliminize(text=watermark)	
	g.export( str(Path(filename).with_suffix('.S8_Om.pdf')) )


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--mcmc-chain', help='emcee chain output')
	parser.add_argument('--param-plot', help='output PDF file', required=True)
	parser.add_argument('--watermark', default=None)
	parser.add_argument('--truth', default=False, action='store_true')
	parser.add_argument('--multinest-dirs', nargs='*', help='multinest output directory')
	parser.add_argument('--labels', nargs='*')
#	parser.add_argument('--planck-chain', help='path to Planck chain (in getdist format)')

	args = parser.parse_args()


	## plot chain.txt to check for convergence of posterior

	if args.mcmc_chain is not None:

		chain = np.loadtxt(args.mcmc_chain)
		burn_in_samples = 10000

		for i in range(1, chain.shape[1]):
			param_chain = chain[:,i]
			walkers = chain[:,0]

			nwalkers = int(walkers.max()) + 1

			plt.figure()

			for j in range(nwalkers):
				this_walker_chain = param_chain[walkers == j]
				plt.plot(this_walker_chain, color='black', alpha=0.2)

			plt.ylabel('parameter value')
			plt.xlabel('step number')
			plt.tight_layout()
			plt.savefig(f"./{args.mcmc_chain}.param_{i}.pdf")
			plt.close()


		## plot posterior projections

		samples = chain[burn_in_samples:, 1:]
		plot_posterior(samples, filename=args.param_plot, watermark=args.watermark)


	if args.multinest_dirs is not None:

		posteriors = []

		for multinest_dir, multinest_label in zip(args.multinest_dirs, args.labels):

			n_dims = 14
			multinest_samples = np.loadtxt(multinest_dir + '.txt')
			multinest_weights = multinest_samples[:, 0]
			multinest_lnL = multinest_samples[:, 1]
			multinest_params = multinest_samples[:, 2:2+n_dims]

			posteriors.append( (multinest_params, multinest_weights,
								utf8tolatex(multinest_label)) )

		plot_posteriors(posteriors,	filename=args.param_plot, watermark=args.watermark,
						plot_truth=args.truth)
