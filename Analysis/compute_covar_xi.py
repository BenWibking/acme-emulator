import numpy as np
import numpy.linalg as linalg
import scipy.integrate as integrate
import scipy.special as special
import configparser
import sys

from compute_covar_DeltaSigma import supersample_and_extend_pk, cov_signal_restrict_scales

def measurement_bins(rmin=0.5,rmax=30.0,nbins=30):
    bins = np.logspace(np.log10(rmin),np.log10(rmax),nbins+1)
    binmin = bins[0:-1]
    binmax = bins[1:]
    assert binmin.shape == binmax.shape
    return list(zip(binmin, binmax))

def j1(x):
    return ( (np.sin(x)/x**2) - (np.cos(x)/x) )

def bin_avg_spherical_j0(k,rminus,rplus):
    """compute the bin-averaged spherical Bessel function j0."""
    integral = lambda r: r**2 * j1(k*r) / k
    return (3.0 / (rplus**3 - rminus**3)) * (integral(rplus) - integral(rminus))

def clustering_signal(k,pk_gg,bins):
    """compute the integral of bessel function
    over the galaxy power spectrum to obtain the 3d real-space correlation function."""
    xi = np.empty(len(bins))
    
    for i, (rminus, rplus) in enumerate(bins):
        # compute signal in bin i on the interval [rminus, rplus)
        y = k**2 / (2.0*np.pi**2) * bin_avg_spherical_j0(k,rminus,rplus) * pk_gg
        result = integrate.simps(y, x=k)
        xi[i] = result

    return xi

def clustering_covariance(k,pk_gg,bins,n_gal=1.0e-3,R=1275.0,L_W=500.):
    """compute the integral of bessel functions
    over the galaxy power spectrum:
    (2.0 / Vol) * int (dk k**2 / 2pi**2) j0(k r_p,i) j0(k r_p,j) (P_gal(k) + 1/n_gal)**2
    j0 must be bin-averaged in order to get results that aren't nonsense.
    See Xu et al. 2013 (eq. 36) MNRAS 431, 2834-2860 (2013)"""

    vol = np.pi * R**2 * L_W

    cov = np.empty((len(bins),len(bins)))
    for i, (r_iminus, r_iplus) in enumerate(bins):
        for j, (r_jminus, r_jplus) in enumerate(bins):
            if j <= i:
                # compute covariance from P(k)
                y = k**2 / (2.0*np.pi**2) * bin_avg_spherical_j0(k,r_iminus,r_iplus) * \
                    bin_avg_spherical_j0(k,r_jminus,r_jplus) * \
                    (pk_gg + 1.0/n_gal)**2
                result = integrate.simps(y, x=k)

                cov[i,j] = result * 2.0/vol
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

def fractional_covariance(binmin,binmax,xi_input,fractional_err):
    """compute the covariance matrix for xi, assuming constant fractional sigma_ii."""
    diag_err = xi_input * fractional_err
    cov = np.diag(diag_err**2)
    return cov

def main(parameter_file, pk_gg_filename,
         output_clustering_filename, binfile, fractional_error=0.):
    
    k_gg_in,pk_gg_in = np.loadtxt(pk_gg_filename,unpack=True)
    k_gg,pk_gg = supersample_and_extend_pk(k_gg_in,pk_gg_in)

    # load parameters
    myparser = configparser.ConfigParser()
    myparser.read(parameter_file)
    params = myparser['params']
    n_gal = float(params['n_gal'])
    R = float(params['R_survey'])
    L_W = float(params['L_W'])
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
    binmin,binmax,_,xi_input = np.loadtxt(binfile,unpack=True)
    bins = list(zip(binmin,binmax))
    if fractional_error == 0.:
        clustering_cov = clustering_covariance(k_gg,pk_gg,bins,n_gal=n_gal,R=R,L_W=L_W)
    else:
        print("computing fractional covariance...")
        clustering_cov = fractional_covariance(binmin,binmax,xi_input,fractional_error)
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
    bin_min, bin_max = zip(*bins)
    bin_centers = 0.5*(np.array(bin_min) + np.array(bin_max))
    np.savetxt(output_clustering_filename, clustering_cov)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('parameter_file',help='parameter filename')
    parser.add_argument('pk_gg_filename',help='name of ASCII input matter power spectrum')
    parser.add_argument('output_clustering_covariance')
    parser.add_argument('--binfile',default=None,help='file with tabulated radial binning')
    parser.add_argument('--fractional_error',type=float,default=0.)
    args = parser.parse_args()

    main(args.parameter_file, args.pk_gg_filename,
         args.output_clustering_covariance,args.binfile,
         fractional_error = args.fractional_error)

