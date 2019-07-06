import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate


def compute_pk(binmin, binmax, xi, kmin=1.e-4, kmax=30.):
    """compute P(k) from tabulated xi(r)."""
    r = 0.5 * (binmin + binmax)  # this should be the bin center
    #    print(1./binmin[0], np.pi/binmin[0]) # this is the (approximate) interval in which the power spectrum goes negative
    dr = binmax - binmin
    dr_over_r = np.min(dr / r)
    logrange = np.log(kmax) - np.log(kmin)
    npoints = int(np.ceil(logrange / (dr_over_r)))
    k = np.logspace(np.log10(kmin), np.log10(kmax), npoints)
    pk = np.zeros(npoints)  # should be resolved s.t. dk/k <~ dr/r

    for i, this_k in enumerate(k):
        integrand = xi * (np.sin(this_k * r) / (this_k * r)) * r**2
        pk[i] = integrate.simps(integrand, x=r)

    pk *= (4.0 * np.pi)
    return k, pk


def pk_from_files(filename, output_file, linear_pk_file, adjust_bias=True):
    binmin, binmax, null, xi = np.loadtxt(filename, unpack=True)
    k, pk = compute_pk(binmin, binmax, xi)

    # define scale on which the linear power spectrum is accurate enough
    lin_scale = np.pi / binmax[-1]

    ## graft linear power spectrum onto large scales
    link, linpk = np.loadtxt(linear_pk_file, unpack=True)
    # construct interpolant for linear pk
    lin_interp = interpolate.interp1d(link, linpk, kind='quadratic')
    # construct interpolant for nonlinear pk
    nl_interp = interpolate.interp1d(k, pk, kind='quadratic')
    # compute linear galaxy bias at pi/rmax (actually also incorporates growth factor ratio)
    if adjust_bias:
        lin_bias = nl_interp(lin_scale) / lin_interp(lin_scale)
    else:
        lin_bias = 1.0
    # interpolate biased linear pk onto (k,pk)_{nonlinear}
    pk[k <= lin_scale] = lin_bias * lin_interp(k[k <= lin_scale])

    np.savetxt(output_file, np.c_[k, pk], delimiter='\t')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('linear_pk')
    parser.add_argument(
        '--dont_adjust_bias', default=False, action='store_true')
    args = parser.parse_args()

    pk_from_files(
        args.input_file,
        args.output_file,
        args.linear_pk,
        adjust_bias=(not args.dont_adjust_bias))
