import numpy as np
import numpy.linalg as linalg
import scipy.integrate as integrate
import scipy.special as special
import configparser

def measurement_bins(rmin=0.5,rmax=30.0,nbins=30):
    bins = np.logspace(np.log10(rmin),np.log10(rmax),nbins+1)
    widths = np.diff(bins)
    centers = bins[0:-1] + 0.5*widths
    return list(zip(centers, widths))

def covariance(bins,fractional_uncertainty=None):
    cov = np.zeros((len(bins),len(bins)))

    # compute normalization
    for i, (r_i, dr_i) in enumerate(bins):
        for j, (r_j, dr_j) in enumerate(bins):
            if j == i:
                cov[i,j] = fractional_uncertainty**2

    return bins,cov

def main(parameter_file, output_cov_filename, observable):
    # load parameters
    myparser = configparser.ConfigParser()
    myparser.read(parameter_file)
    params = myparser['params']
    if observable == 'wp':
        frac_err_wp = float(params['frac_err_wp'])
        fractional_uncertainty = frac_err_wp
        bins = measurement_bins()
    elif observable == 'DS':
        frac_err_DS = float(params['frac_err_DS'])
        fractional_uncertainty = frac_err_DS
        bins = measurement_bins()
    elif observable == 'ngal':
        frac_err_ngal = float(params['frac_err_ngal'])
        fractional_uncertainty = frac_err_ngal
        bins = measurement_bins(nbins=1)

    ## covariance
    bins,cov = covariance(bins,fractional_uncertainty=fractional_uncertainty)
    bin_centers, bin_widths = zip(*bins)

    np.savetxt(output_cov_filename, cov)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('parameter_file',help='parmeter filename')
    parser.add_argument('output_covariance')
    parser.add_argument('--observable',choices=['wp','DS','ngal'],required=True)
    args = parser.parse_args()

    main(args.parameter_file, args.output_covariance, args.observable)

