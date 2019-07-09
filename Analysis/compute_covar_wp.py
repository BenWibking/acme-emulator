import numpy as np
import numpy.linalg as linalg
import scipy.integrate as integrate
import scipy.special as special
import configparser

from compute_covar_DeltaSigma import supersample_and_extend_pk, measurement_bins, my_sinc, window, cov_signal_restrict_scales

def clustering_signal(k,pk_gg,bins):
    """compute the integral of bessel function
    over the galaxy power spectrum to obtain the projected correlation function."""
    #bins = measurement_bins()
    wp = np.empty(len(bins))
    
    for i, (r, dr) in enumerate(bins):
        # compute signal in bin i centered at r with width dr
        y = k / (2.0*np.pi) * special.j0(k*r) *\
            my_sinc(k*dr/2.0) * \
            (pk_gg)
        result = integrate.simps(y, x=k)
        wp[i] = result

    return wp

def clustering_covariance(k,pk_gg,bins,n_gal=1.0e-3,R=1275.0):
    """compute the integral of bessel functions
    over the galaxy power spectrum:
    int (dk k / 2pi) J_0(k r_p,i) J_0(k r_p,j) (P_gal(k) + 1/n_gal)**2
    A separate integral must be done for normalization
    (that integral depends on the galaxy survey properties).

    One must damp the high-k limit in order to get results that aren't nonsense."""
    #bins = measurement_bins()
    cov = np.empty((len(bins),len(bins)))

    # compute normalization
    for i, (r_i, dr_i) in enumerate(bins):
        for j, (r_j, dr_j) in enumerate(bins):
            if j <= i:
                # compute covariance from P(k)
                y = k / (2.0*np.pi) * special.j0(k*r_i) * special.j0(k*r_j) * \
                    my_sinc(k*dr_i/2.0) * my_sinc(k*dr_j/2.0) * \
                    (pk_gg + 1.0/n_gal)**2
                result = integrate.simps(y, x=k)

                # compute normalization
                norm_ij = k / (2.0*np.pi) * special.j0(k*r_i) * special.j0(k*r_j)*\
                          my_sinc(k*dr_i/2.0) * my_sinc(k*dr_j/2.0) * \
                          window(k,R=R)**2
                norm_i = k / (2.0*np.pi) * special.j0(k*r_i) * my_sinc(k*dr_i/2.0) * window(k,R=R)**2
                norm_j = k / (2.0*np.pi) * special.j0(k*r_j) * my_sinc(k*dr_j/2.0) * window(k,R=R)**2
                A_ij = integrate.simps(norm_ij, x=k)
                A_i = integrate.simps(norm_i, x=k)
                A_j = integrate.simps(norm_j, x=k)

                cov[i,j] = result * 2.0*A_ij / (A_i*A_j)
                cov[j,i] = cov[i,j]

    return cov

def compute_signal_to_noise(cov,signal):
    """compute the signal-to-noise for the fiducial case."""
    SN = np.sqrt(np.dot(signal, np.dot(np.linalg.inv(cov), signal)))
    return SN

def cleaned_precision_matrix(cov):
    """zero-out noisy modes"""
    U, s, V = linalg.svd(cov, full_matrices=True)
    S = np.diag(s)
    S_inv = np.diag(1.0/s)
    noise_threshold_inv = 1.0e6 # singular value threshold
    noise_threshold = 1.0/noise_threshold_inv
    S_inv[S_inv >= noise_threshold_inv] = 0.
    S[S <= noise_threshold] = 0.
    cov_inv_clean = np.dot(U, np.dot(S_inv, V))
    cov_clean = np.dot(U, np.dot(S, V))
    return cov_clean, cov_inv_clean

def main(parameter_file, 
         pk_gg_filename, output_clustering_filename, output_clustering_precision, output_clustering_signal):
    import sys

    k_gg_in,pk_gg_in = np.loadtxt(pk_gg_filename,unpack=True)
    k_gg,pk_gg = supersample_and_extend_pk(k_gg_in,pk_gg_in)

    # load parameters
    myparser = configparser.ConfigParser()
    myparser.read(parameter_file)
    params = myparser['params']
    n_gal = float(params['n_gal'])
    R = float(params['R_survey'])
    # scale-restriction parameters
    rp_min = 0.
    rp_max = np.inf
    do_rescale = 'False'
    rp_min_fiducial = 0.
    if 'wp_rp_min' in params:
        rp_min = float(params['wp_rp_min'])
    if 'wp_rp_max' in params:
        rp_max = float(params['wp_rp_max'])
    if 'wp_do_rescale' in params:
        do_rescale = params['wp_do_rescale']    
        rp_min_fiducial = float(params['wp_rp_min_fiducial'])
        print("\tdo_rescale = {}".format(do_rescale),file=sys.stderr)
        print("\trp_min_fiducial = {}".format(rp_min_fiducial),file=sys.stderr)

    ## clustering covariance
    projected_rmin = 0.5
    projected_rmax = 30.0
    projected_nbins = 30
    if 'projected_rmin' in params:
        projected_rmin = float(params['projected_rmin'])
    if 'projected_rmax' in params:
        projected_rmax = float(params['projected_rmax'])
    if 'projected_nbins' in params:
        projected_nbins = float(params['projected_nbins'])

    bins = measurement_bins(rmin=projected_rmin,rmax=projected_rmax,nbins=projected_nbins)
    clustering_cov = clustering_covariance(k_gg,pk_gg,bins,n_gal=n_gal,R=R)
    wp = clustering_signal(k_gg,pk_gg,bins)

    ## compute SN_rescaled from restricting cov to wp_rp_min_rescale < rp < np.inf
    cov_fiducial_scales, wp_fiducial_scales = cov_signal_restrict_scales(clustering_cov, wp, bins, rp_min=rp_min_fiducial)
    SN_fiducial_scales = compute_signal_to_noise(cov_fiducial_scales, wp_fiducial_scales)
    print('\tsignal-to-noise (fiducial scales): %s' % SN_fiducial_scales, file=sys.stderr)

    if 'wp_rp_min' in params or 'wp_rp_max' in params:
        print("\trp_min = {} rp_max = {}".format(rp_min,rp_max),file=sys.stderr)
        cov_mask, wp_mask = cov_signal_restrict_scales(clustering_cov, wp, bins, rp_min=rp_min, rp_max=rp_max)
        SN_mask = compute_signal_to_noise(cov_mask, wp_mask)
        print('\tsignal-to-noise (scale-restricted): %s' % SN_mask, file=sys.stderr)

    if do_rescale == 'True':
        rescale_factor = (SN_mask/SN_fiducial_scales)**2
        print("\trescale factor = {}".format(rescale_factor), file=sys.stderr)
        clustering_cov *= rescale_factor # rescale output covariance matrix
        cov_mask *= rescale_factor
        print('\tsignal-to-noise (rescaled): %s' % compute_signal_to_noise(cov_mask, wp_mask),file=sys.stderr)

    ## output full covariance matrix (all computed scales); fisher code will exclude bins if needed
    cov_clean, cov_inv_clean = cleaned_precision_matrix(clustering_cov)
    bin_centers, bin_widths = zip(*bins)
    np.savetxt(output_clustering_filename, clustering_cov)
    np.savetxt(output_clustering_precision, cov_inv_clean)
    np.savetxt(output_clustering_signal, np.c_[bin_centers, wp])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('parameter_file',help='parmeter filename')
    parser.add_argument('pk_gg_filename',help='name of ASCII input matter power spectrum')
    parser.add_argument('output_clustering_covariance')
    parser.add_argument('output_clustering_precision')
    parser.add_argument('output_clustering_signal')
    args = parser.parse_args()

    main(args.parameter_file, args.pk_gg_filename,
         args.output_clustering_covariance, args.output_clustering_precision,
         args.output_clustering_signal)

