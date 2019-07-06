import numpy as np
import scipy.integrate as integrate


def elementwise_integral_firstorder(rp,binmin,binmax,xi,pimax):

    lower_bound = rp
    upper_bound = np.sqrt(rp**2 + pimax**2)

    binmask = np.logical_and(binmax > lower_bound, binmin < upper_bound)
    masked_xi = xi[binmask]
    r_i = binmin[binmask]
    r_iplus = binmax[binmask]
    s_plus = np.minimum(upper_bound, r_iplus)
    s_minus = np.maximum(lower_bound, r_i)
    
    # here we assume that xi is piecewise constant over the tabulated input bins
    integral = 2.0*masked_xi * \
               (np.sqrt(s_plus**2 - rp**2) - np.sqrt(s_minus**2 - rp**2))
               
    return np.sum(integral)


def elementwise_integral_secondorder(rp,binmin,binmax,xi,pimax):

    lower_bound = rp
    upper_bound = np.sqrt(rp**2 + pimax**2)

    # offset bins by 0.5*dr
    bin_median = 0.5*(binmin+binmax)
    bin_minus = bin_median[:-1]
    bin_plus = bin_median[1:]
    binmask = np.logical_and(bin_plus > lower_bound, bin_minus < upper_bound)
    xi_minus = xi[:-1][binmask]
    xi_plus = xi[1:][binmask]
    r_minus = bin_minus[binmask]
    r_plus = bin_plus[binmask]
    
    # integration limits may lie within a bin, need to be careful
    s_minus = np.maximum(lower_bound, r_minus)
    s_plus = np.minimum(upper_bound, r_plus)

    # here we assume that xi is piecewise linear over the tabulated input bins
    
    m = (xi_plus - xi_minus) / (r_plus - r_minus)
    const_term = 2.0*(xi_minus - m*r_minus) * \
                 (np.sqrt(s_plus**2 - rp**2) - np.sqrt(s_minus**2 - rp**2))
    linear_term = m * ( s_plus*np.sqrt(s_plus**2 - rp**2) - \
                        s_minus*np.sqrt(s_minus**2 - rp**2) + \
                        rp**2 * np.log( (s_plus + np.sqrt(s_plus**2 - rp**2)) / \
                                         (s_minus + np.sqrt(s_minus**2 - rp**2)) ) )
    integral = linear_term + const_term
    
    return np.sum(integral)


def wp(binmin,binmax,xi,Omega_m,z_lens,pimax=100.,Omega_m_fid=0.3,
       rp_min=0.5, rp_max=30.0, nbins=30):
       
    """compute w_p(r_p) from tabulated xi(r)."""

    rp_bins = np.logspace(np.log10(rp_min), np.log10(rp_max), nbins+1)
    rp_binmin = rp_bins[0:-1]
    rp_binmax = rp_bins[1:]
    w_p = np.zeros(rp_binmin.shape[0])

    # compute comoving distance ratios in true/fiducial cosmology
    E = lambda z: 1.0/np.sqrt(Omega_m*(1.0+z)**3 + (1.0-Omega_m))
    E_fid = lambda z: 1.0/np.sqrt(Omega_m_fid*(1.0+z)**3 + (1.0-Omega_m_fid))
    Rc_true = integrate.quad(E,0.,z_lens)[0]
    Rc_fid = integrate.quad(E_fid,0.,z_lens)[0]

    for i,(this_rp_binmin,this_rp_binmax) in enumerate(zip(rp_binmin,rp_binmax)):
    
        rp_fid = 0.5*(this_rp_binmin + this_rp_binmax)
        rp = (Rc_true/Rc_fid)*rp_fid # convert to true rp in this cosmology
        w_p[i] += elementwise_integral_secondorder(rp,binmin,binmax,xi,pimax)

    return rp_binmin, rp_binmax, w_p


def wp_from_files(header_file,filename,output_file,pimax,z_lens,rp_min,rp_max,nbins):

    # read in cosmological parameters from header_file

    import config
    cf = config.AbacusConfigFile(header_file)
    omega_m = cf.Omega_M # at z=0

    binmin, binmax, _, xi = np.loadtxt(filename,unpack=True)
    
    rp_binmin, rp_binmax, w_p = wp(binmin, binmax, xi,
    							   omega_m, float(z_lens), pimax=float(pimax),
                                   rp_min=float(rp_min), rp_max=float(rp_max),
                                   nbins=int(nbins))
    
    np.savetxt(output_file, np.c_[rp_binmin, rp_binmax, np.zeros(w_p.shape[0]), w_p],
               delimiter='\t')


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_file')
    parser.add_argument('header_file')
    parser.add_argument('output_file')

    parser.add_argument('--pimax',default=100.)
    parser.add_argument('--zlens',default=0.27)

    parser.add_argument('--rpmin',default=0.5)
    parser.add_argument('--rpmax',default=30.0)
    parser.add_argument('--nbins',default=30)

    args = parser.parse_args()

    wp_from_files(args.header_file,
                  args.input_file, args.output_file, args.pimax, args.zlens,
                  args.rpmin, args.rpmax, args.nbins)
