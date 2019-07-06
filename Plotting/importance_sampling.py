import numpy as np
import argparse
import getdist


def do_importance_sampling(new_cinv, new_datavector, modelvectors):
	
	"""do importance sampling on 'samples' with Gaussian likelihood with covariance 'new_cov'.
		We also want to re-use the model predictions for all of the samples,
		since we're only interested in modifying the covariance used in our likelihood
		(or rescaling the model predictions by a constant) or replacing the datavector."""
		
	new_loglikes = np.empty(modelvectors.shape[0])
	
	for i in range(modelvectors.shape[0]):
		y = new_datavector - modelvectors[i,:]
		new_loglikes[i] = -0.5 * ( y.T @ new_cinv @ y ) # don't bother normalizing for now
	
	return new_loglikes


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	
	parser.add_argument('--multinest-dir', required=True, help='multinest output directory')
	parser.add_argument('inv_cov_file')
	parser.add_argument('cov_file')
	parser.add_argument('output_dir')
	
#	parser.add_argument('--datavector-wp', required=True)
#	parser.add_argument('--datavector-ds', required=True)
#	parser.add_argument('--rp-min', required=True)

	args = parser.parse_args()


	## read in samples
	
	n_dims = 14
	
	multinest_samples = np.loadtxt(args.multinest_dir + '.txt')
	multinest_weights = multinest_samples[:, 0]
	multinest_lnL = multinest_samples[:, 1]		# this is actually -2.0*lnL !
	multinest_params = multinest_samples[:, 2:2+n_dims]
	multinest_ppd = multinest_samples[:, 2+n_dims:]
		
	multinest_chisqs = multinest_ppd[:, -1]
	multinest_modelvecs = multinest_ppd[:, :-1]
	multinest_wps, multinest_deltasigmas = np.split(multinest_modelvecs, 2, axis=1)
	
	samples = getdist.MCSamples(samples=multinest_samples,
								weights=multinest_weights,
								loglikes=multinest_lnL)
	

	## read datavector

	rmin_mock, rmax_mock, _, wp_mock = np.loadtxt('../../lowz_mocks/data/lowz_corr.wp.txt',
													unpack=True)
	rmin_mock, rmax_mock, _, ds_mock = np.loadtxt('../../lowz_mocks/data/lowz_corr.ds.txt',
													unpack=True)
#	rmin_mock, rmax_mock, _, wp_mock = np.loadtxt('../../lowz_mocks/data/lowz_corr_blinded.wp.txt',
#													unpack=True)
#	rmin_mock, rmax_mock, _, ds_mock = np.loadtxt('../../lowz_mocks/data/lowz_corr_blinded.ds.txt',
#													unpack=True)

	r_mock = 0.5*(rmin_mock + rmax_mock)

#	rmin_cut = 0.6		## TODO: make this a parameter
	rmin_cut = 2.0		## TODO: make this a parameter
	scale_mask = rmin_mock > rmin_cut
	
	datavector = np.concatenate( [wp_mock[scale_mask], ds_mock[scale_mask]] )
	modelvectors = np.hstack( [multinest_wps[:, scale_mask], multinest_deltasigmas[:, scale_mask]] )


	## read covariance matrix

	Cinv = np.loadtxt(args.inv_cov_file)	# inverse covariance matrix
	cov = np.loadtxt(args.cov_file)			# covariance matrix


	## compute importance sampling
	
	lnL = do_importance_sampling(Cinv, datavector, modelvectors)
	
	prior_mass = multinest_weights / np.exp(multinest_lnL/(-2.0))	# unnormalized
	weights = prior_mass * np.exp(lnL)								# unnormalized
	chisqs = -2.0*lnL												# unnormalized
	
	ppd = np.c_[multinest_modelvecs, chisqs]
	params = multinest_params

	print(f"Mean delta chi^2 = {np.mean(chisqs - multinest_chisqs):.3f}")
	
	np.savetxt(args.output_dir + '.txt', np.c_[weights, -2.0*lnL, params, ppd])
	

	## compute equal weighted samples
	
	norm = np.sum(weights)
	weights *= 1.0/norm	
	nsamples = 5000
	candidate_samples = np.c_[params, ppd, lnL]
	chosen_samples = candidate_samples[np.random.choice(candidate_samples.shape[0], size=nsamples, replace=False, p=weights)]
	
	np.savetxt(args.output_dir + 'post_equal_weights.dat', chosen_samples)