import numpy as np
import numpy.linalg as linalg
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.special as special
import configparser
from compute_DeltaSigma import sigma_crit


def cov_signal_restrict_scales(cov, signal, bins, rp_min=0., rp_max=np.inf):
	"""return cov, wp restricted to scales rmin < rp < rmax"""
	bin_centers, bin_widths = zip(*bins)
	bin_centers = np.array(bin_centers)
	binmask = np.logical_and(bin_centers > rp_min, bin_centers < rp_max)
	cov_mask = cov[binmask, :][:, binmask]
	signal_mask = signal[binmask]
	return cov_mask, signal_mask


def measurement_bins(rmin=0.5, rmax=30.0, nbins=30):
	bins = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
	widths = np.diff(bins)
	centers = bins[0:-1] + 0.5 * widths
	return list(zip(centers, widths))


def my_sinc(x):
	return np.sin(x) / x


def supersample_pk(k, pk, supersample_factor=10):
	"""supersample pk by cubic spline interpolation"""
	interpolator = interpolate.interp1d(k, pk, kind='cubic')
	newk = np.logspace(
		np.log10(k[0]), np.log10(k[-1]), supersample_factor * k.shape[0])
	newpk = interpolator(newk)
	return newk, newpk


def extend_pk_range(k, pk, extended_kmax=1.0e3):
	"""extend tabulated input power spectrum pk(k) to k_max using k^-3 asymptotic expansion"""
	kmin = k[0]
	kmax = k[-1]
	log_krange = np.log10(kmax / kmin)
	bins_per_log_k = k.shape[0] / log_krange
	asymp_log_krange = np.log10(extended_kmax / kmax)
	asymp_k = np.logspace(
		np.log10(kmax), np.log10(extended_kmax),
		int(asymp_log_krange * bins_per_log_k))[1:]
	asymp_pk = pk[-1] * (asymp_k / kmax)**(-3)
	extended_k = np.concatenate((k, asymp_k))
	extended_pk = np.concatenate((pk, asymp_pk))
	return extended_k, extended_pk


def supersample_and_extend_pk(k, pk):
	this_k, this_pk = supersample_pk(k, pk)
	return extend_pk_range(this_k, this_pk)


def survey_radius(degrees_sq, redshift, Omega_m=0.3):
	"""compute the comoving survey radius assuming a spherical cap survey of
		'degrees_sq' square degrees on the sky at redshift 'redshift'."""

	sr = 4.0*np.pi * (degrees_sq / 41253.)	# steradians
	R = comoving(redshift, Omega_m=Omega_m)
	#	area = sr * R**2
	#	theta = np.arccos( 1.0 - (area / (2.0*np.pi*R**2)) )
	theta = np.arccos( 1.0 - (sr / (2.0*np.pi)) )
	R_survey = R*theta

	return R_survey


def comoving(redshift, Omega_m=0.3):
	"""compute comoving radial distance to redshift 'redshift' in flat LCDM."""
	norm_fac = 3000.0 # Mpc/h
	E = lambda z: Omega_m*(1+z)**3 + (1-Omega_m)	# ignore radiation, neutrinos, etc.
	integrand = lambda z: 1.0/np.sqrt(E(z))
	chi = norm_fac * integrate.quad(integrand, 0., redshift)[0]	# Mpc/h units
	return chi


def pi_lens(z_lens, z_source, Omega_m=0.3):
	"""compute pi_lens (eq. 18 in my notes) assuming a single source and single lens redshift."""
	norm_fac = 3000.0 # Mpc/h (comoving)
	sigma_crit_fid = sigma_crit(z_lens, z_source, Omega_m=Omega_m)
	E = lambda z: Omega_m*(1+z)**3 + (1-Omega_m)	# ignore radiation, neutrinos, etc.
	integrand = lambda z: (sigma_crit_fid / sigma_crit(z, z_source, Omega_m=Omega_m))**2 * \
							 E(z)**(-1/2)
	pi = norm_fac * integrate.quad(integrand, 0., z_source)[0]
	return pi


def window(k, R=1275.0):
	"""this is the window function for a circular survey of projected comoving radius R"""
	return (2.0 * np.pi * R**2) * special.j1(k * R) / (k * R)


def lensing_signal(k, pk_gm, bins, critical_density, Omega_M=0.3, H_0=100., gamma_t=True):
	"""compute the integral of bessel functions
	over the galaxy power spectrum AND the matter power spectrum
	to obtain the tangential excess shear signal."""

	delta_sigma = np.empty(len(bins))

	# compute mean_rho (comoving density units = Msun pc^-3)
	speed_of_light_km_s = 2.998e5  # km/s
	csq_over_G = 2.494e12  # 3c^2/(8*pi*G) Msun pc^-1
	mean_rho = Omega_M * csq_over_G * (H_0 / speed_of_light_km_s)**2 / 1.0e12

	for i, (r, dr) in enumerate(bins):
		y2 = k / (2.0*np.pi) * special.jn(2,k*r) * \
		my_sinc(k*dr/2.0) * \
		(mean_rho*pk_gm) * 1.0e6
		# Mpc^-2 Msun pc^-3 Mpc^3 = Msun pc^-3 Mpc^3 Mpc^-2 = Msun pc^-3 Mpc (pc/Mpc)
		result2 = integrate.simps(y2, x=k)
		delta_sigma[i] = result2

#	if gamma_t == True:
#		gamma_t = delta_sigma / critical_density
#		return gamma_t
		
	return delta_sigma


def lensing_covariance(k,
					Pk_gm,
					Pk_gg,
					Pk_mm,
					bins,
					n_gal=1.0e-3,
					Omega_M=0.3,
					H_0=100.,
					L_W=500.,
					DeltaPi=400.,
					critical_density=4.7e3,
					n_source=8.0,
					shape_noise=0.21,
					R=1275.0,
					gamma_t=True):
					
	"""compute the integral of Bessel functions
	over the galaxy power spectrum AND the matter power spectrum:
	int (dk k / 2pi) J_2(k r_p,i) J_2(k r_p,j) (P_gal(k) + 1/_ngal)(P_shear(k) + const.)"""

	cov = np.empty((len(bins), len(bins)))
	cov_nointrinsic = np.empty((len(bins), len(bins)))

	## compute mean_rho (comoving density units = Msun pc^-3)
	speed_of_light_km_s = 2.998e5  # km/s
	csq_over_G = 2.494e12  # 3c^2/(8*pi*G) Msun pc^-1
	mean_rho = Omega_M * csq_over_G * (
		H_0 / speed_of_light_km_s)**2 / 1.0e12  # Msun pc^-3


	## compute shape noise term (surface density: Msun pc^-2; shear signal: Msun pc^-2)
	# z_lens = 0.27 # (unused) in the future, use this to compute the critical density
	# L_W = 500. # h^-1 Mpc (the redshift range of the lenses)
	# DeltaPi = 400. # h^-1 Mpc (approximately the redshift range of the lenses)
	# critical_density = 4.7e3 # h Msun pc^-2
	# n_source = 8.0 # h^2 Mpc^-2 (source galaxy projected number density)
	# shear_responsivity = 0.85
	# shape_noise = 0.36 / (2.0*shear_responsivity)

	shape_noise_term = critical_density**2 * (shape_noise**2 / n_source)
						# (Msun pc^-2)^2 Mpc^2
	
	## NOTE: this is an *approximate* form of the projected shear-shear power spectrum
	## [see Singh+ 2017 eq. A34 for the "exact" (Limber-approximated) form.]

	shear_term = DeltaPi * mean_rho**2 * Pk_mm * 1.0e12
						# (Msun pc^-3)^2 Mpc^4 (pc/Mpc)^2

	## split into 'simulation' and 'LSS' terms

	DeltaPi_sim = 200. # Mpc/h
	shear_sim = DeltaPi_sim * mean_rho**2 * Pk_mm * 1.0e12

	DeltaPi_lss = (DeltaPi - DeltaPi_sim)
	shear_lss = DeltaPi_lss * mean_rho**2 * Pk_mm * 1.0e12
	
	# = Msun^2 pc^-4 Mpc^2
	#	(convert Mpc^2 -> pc^2: Msun^2 pc^-6 pc^2 Mpc^2 = Msun^2 pc^-4 Mpc^2)

	for i, (r_i, dr_i) in enumerate(bins):
		for j, (r_j, dr_j) in enumerate(bins[:i + 1]):
		
			# compute shape noise part
			y0 = k / (2.0*np.pi) * special.jn(2,k*r_i) * special.jn(2,k*r_j) * \
					my_sinc(k*dr_i/2.0) * my_sinc(k*dr_j/2.0) * \
					(Pk_gg + 1.0/n_gal) * shape_noise_term

			shape = integrate.simps(y0, x=k)

			# compute LSS part
			y1 = k / (2.0*np.pi) * special.jn(2,k*r_i) * special.jn(2,k*r_j) * \
					my_sinc(k*dr_i/2.0) * my_sinc(k*dr_j/2.0) * \
					( (Pk_gg + 1.0/n_gal)*shear_lss + (mean_rho*Pk_gm)**2 * (DeltaPi_lss*1.0e12) )

			lss = integrate.simps(y1, x=k)

			# compute 'intrinsic' part
			y2 = k / (2.0*np.pi) * special.jn(2,k*r_i) * special.jn(2,k*r_j) * \
					my_sinc(k*dr_i/2.0) * my_sinc(k*dr_j/2.0) * \
					( (Pk_gg + 1.0/n_gal)*shear_sim + (mean_rho*Pk_gm)**2 * (DeltaPi_sim*1.0e12) )

			# Mpc^-3 Mpc^-2 Mpc^6 Msun^2 pc^-6 Mpc = Mpc^2 Msun^2 pc^-6 = Msun^2 pc^-4 * 1.0e12

			intrinsic = integrate.simps(y2, x=k)

			# compute survey window effects on effective volume
			norm_ij = k / (2.0*np.pi) * special.j0(k*r_i) * special.j0(k*r_j) *\
						my_sinc(k*dr_i/2.0) * my_sinc(k*dr_j/2.0) * window(k,R=R)**2
			norm_i = k / (2.0 * np.pi) * special.j0(k * r_i) *\
						my_sinc(k * dr_i / 2.0) * window(k, R=R)**2
			norm_j = k / (2.0 * np.pi) * special.j0(k * r_j) *\
						my_sinc(k * dr_j / 2.0) * window(k, R=R)**2
			
			V_ij = integrate.simps(norm_ij, x=k) * L_W
			V_i = integrate.simps(norm_i, x=k) * L_W
			V_j = integrate.simps(norm_j, x=k) * L_W
			V_eff = (V_i * V_j) / V_ij

			cov[i, j] = (shape + lss + intrinsic) / V_eff
			cov_nointrinsic[i, j] = (shape + lss) / V_eff

			cov[j, i] = cov[i, j]
			cov_nointrinsic[j, i] = cov_nointrinsic[i, j]

#	if gamma_t == True:
#		cov /= critical_density**2		# compute cov_{\gamma_t}

	return cov, cov_nointrinsic


def cleaned_precision_matrix(cov):
	"""zero-out noisy modes"""
	U, s, V = linalg.svd(cov, full_matrices=True)
	S = np.diag(s)
	S_inv = np.diag(1.0 / s)
	noise_threshold_inv = 1.0e6  # singular value threshold
	noise_threshold = 1.0 / noise_threshold_inv
	S_inv[S_inv >= noise_threshold_inv] = 0.
	S[S <= noise_threshold] = 0.
	cov_inv_clean = np.dot(U, np.dot(S_inv, V))
	cov_clean = np.dot(U, np.dot(S, V))
	return cov_clean, cov_inv_clean


def compute_signal_to_noise(cov, signal):
	"""compute the signal-to-noise for the fiducial case."""
	SN = np.sqrt(np.dot(signal, np.dot(np.linalg.inv(cov), signal)))
	return SN


def main(parameter_file, pk_gg_filename, pk_gm_filename, pk_mm_filename,
		output_lensing_filename, output_lensing_precision,
		output_lensing_signal, output_lensing_nointrinsic_filename, gamma_t=None):

	import sys

	k_gg_in, pk_gg_in = np.loadtxt(pk_gg_filename, unpack=True)
	k_gm_in, pk_gm_in = np.loadtxt(pk_gm_filename, unpack=True)
	k_mm_in, pk_mm_in = np.loadtxt(pk_mm_filename, unpack=True)

	k_gg, pk_gg = supersample_and_extend_pk(k_gg_in, pk_gg_in)
	k_gm, pk_gm = supersample_and_extend_pk(k_gm_in, pk_gm_in)
	k_mm, pk_mm = supersample_and_extend_pk(k_mm_in, pk_mm_in)


	## load parameters
	
	myparser = configparser.ConfigParser()
	myparser.read(parameter_file)
	params = myparser['params']
	
	n_gal 				= float(params['n_gal'])
	Omega_M 			= float(params['Omega_M'])
	L_W 				= float(params['L_W'])
	DeltaPi 			= float(params['DeltaPi'])
	critical_density 	= float(params['critical_density'])
	n_source 			= float(params['n_source'])
	shape_noise 		= float(params['shape_noise'])
	R 					= float(params['R_survey'])


	## scale-restriction parameters
	
	rp_min = 0.
	rp_max = np.inf
	do_rescale = 'False'
	rp_min_fiducial = 0.
	
	if 'DS_rp_min' in params:
		rp_min = float(params['DS_rp_min'])
	if 'DS_rp_max' in params:
		rp_max = float(params['DS_rp_max'])
	if 'DS_do_rescale' in params:
		do_rescale = params['DS_do_rescale']
		rp_min_fiducial = float(params['DS_rp_min_fiducial'])


	## lensing covariance
	
	projected_rmin = 0.5
	projected_rmax = 30.0
	projected_nbins = 30
	
	if 'projected_rmin' in params:
		projected_rmin = float(params['projected_rmin'])
	if 'projected_rmax' in params:
		projected_rmax = float(params['projected_rmax'])
	if 'projected_nbins' in params:
		projected_nbins = float(params['projected_nbins'])

	bins = measurement_bins(
		rmin=projected_rmin, rmax=projected_rmax, nbins=projected_nbins)
		
	lens_cov, lens_cov_nointrinsic = lensing_covariance(
		k_gm,
		pk_gm,
		pk_gg,
		pk_mm,
		bins,
		n_gal=n_gal,
		Omega_M=Omega_M,
		L_W=L_W,
		DeltaPi=DeltaPi,
		critical_density=critical_density,
		n_source=n_source,
		shape_noise=shape_noise,
		R=R,
		gamma_t=gamma_t)
		
	delta_sigma = lensing_signal(
		k_gm, pk_gm, bins, critical_density, Omega_M=Omega_M, gamma_t=gamma_t)


	## compute SN_rescaled from restricting cov to DS_rp_min_rescale < rp < np.inf
	
	cov_fiducial_scales, DS_fiducial_scales = cov_signal_restrict_scales(
		lens_cov, delta_sigma, bins, rp_min=rp_min_fiducial)
	SN_fiducial_scales = compute_signal_to_noise(cov_fiducial_scales,
												DS_fiducial_scales)
												
	print('\tsignal-to-noise (fiducial scales): %s' % SN_fiducial_scales, file=sys.stderr)

	bin_centers, bin_widths = zip(*bins)
	
	if 'DS_rp_min' in params or 'DS_rp_max' in params:
		cov_mask, delta_sigma_mask = cov_signal_restrict_scales(
			lens_cov, delta_sigma, bins, rp_min=rp_min, rp_max=rp_max)
		SN_mask = compute_signal_to_noise(cov_mask, delta_sigma_mask)
		print(
			'\tsignal-to-noise (scale-restricted): %s' % SN_mask,
			file=sys.stderr)

	if do_rescale == 'True':
		rescale_factor = (SN_mask / SN_fiducial_scales)**2
		lens_cov *= rescale_factor
		cov_mask *= rescale_factor
		print(
			'\tsignal-to-noise (rescaled): %s' % compute_signal_to_noise(
				cov_mask, delta_sigma_mask),
			file=sys.stderr)

	np.savetxt(output_lensing_filename, lens_cov)

	if output_lensing_nointrinsic_filename is not None:
		print(f"nointrinsic signal-to-noise (all scales): {compute_signal_to_noise(lens_cov_nointrinsic, delta_sigma)}")
		np.savetxt(output_lensing_nointrinsic_filename, lens_cov_nointrinsic)
	
	if output_lensing_precision is not None:
		lens_cov_clean, lens_cov_inv_clean = cleaned_precision_matrix(lens_cov)
		np.savetxt(output_lensing_precision, lens_cov_inv_clean)
	
	if output_lensing_signal is not None:
		np.savetxt(output_lensing_signal, np.c_[bin_centers, delta_sigma])


if __name__ == "__main__":

	import argparse
	parser = argparse.ArgumentParser()
	
	parser.add_argument('parameter_file', help='parmeter filename')
	
	parser.add_argument(
		'pk_gg_filename', help='name of ASCII input matter power spectrum')
	parser.add_argument(
		'pk_gm_filename', help='name of ASCII input matter power spectrum')
	parser.add_argument(
		'pk_mm_filename', help='name of ASCII input matter power spectrum')
		
	parser.add_argument('output_lensing_covariance')
	parser.add_argument('--output-lensing-precision', default=None)
	parser.add_argument('--output-lensing-signal', default=None)
	parser.add_argument('--output-cov-nointrinsic', default=None)
	
	parser.add_argument('--gamma_t', type=bool, default=True, help='compute gamma_t')
	
	args = parser.parse_args()

	main(args.parameter_file, args.pk_gg_filename, args.pk_gm_filename,
		args.pk_mm_filename, args.output_lensing_covariance,
		args.output_lensing_precision, args.output_lensing_signal,
		args.output_cov_nointrinsic,
		gamma_t=args.gamma_t)
