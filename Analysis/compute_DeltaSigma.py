import numpy as np
import scipy.integrate as integrate
from compute_wp import elementwise_integral_secondorder


def wp(rp, binmin, binmax, xi, pimax):
    """compute wp(r_p) from tabulated xi(r)."""
    return elementwise_integral_secondorder(rp, binmin, binmax, xi, pimax)


def sigma_crit(z_lens, z_source, Omega_m=0.3):
    """compute sigma_crit assuming the redshift distributions are delta functions."""
    D_H = 3000.0  # Mpc/h
    D_H_pc = D_H * 1.0e6  # pc/h
    three_csq_over_8piG = 2.494e12  # 3c^2/(8*pi*G) Msun pc^-1
    csq_over_4piG = three_csq_over_8piG * (2. / 3.)
    Omega_L = 1.0 - Omega_m

    integrand = lambda z: 1.0 / np.sqrt(Omega_m * (1.0 + z)**3 + Omega_L)
    D_lens = D_H_pc * integrate.quad(integrand, 0., z_lens)[0]
    D_source = D_H_pc * integrate.quad(integrand, 0., z_source)[0]

    dist_fac = D_source / ((1.0 + z_lens) * (D_lens * (D_source - D_lens)))
    sigma_c = csq_over_4piG * dist_fac  # h Msun pc^-2
    return sigma_c


def DeltaSigma(binmin,
               binmax,
               xi,
               pimax=None,
               H0=None,
               Omega_m=None,
               Omega_m_fid=0.3,
               z_lens=None,
               z_source=None,
               rp_min=0.5,
               rp_max=30.0,
               nbins=30):
    # mean rho (in comoving Msun pc^-2, no little h)
    h = H0 / 100.
    # compute mean_rho (comoving density units = Msun pc^-3)
    speed_of_light_km_s = 2.998e5  # km/s
    csq_over_G = 2.494e12  # 3c^2/(8*pi*G) Msun pc^-1
    mean_rho = Omega_m * csq_over_G * (
        H0 / speed_of_light_km_s)**2 / 1.0e12  # Msun pc^-3

    # compute rp bins
    rp_bins = np.logspace(np.log10(rp_min), np.log10(rp_max), nbins + 1)
    rp_binmin = rp_bins[0:-1]
    rp_binmax = rp_bins[1:]
    rp_mid_fid = (rp_binmin + rp_binmax) / 2.0

    # compute comoving distance ratios in true/fiducial cosmology
    E = lambda z: 1.0 / np.sqrt(Omega_m * (1.0 + z)**3 + (1.0 - Omega_m))
    E_fid = lambda z: 1.0 / np.sqrt(Omega_m_fid * (1.0 + z)**3 + (1.0 -
                                                                  Omega_m_fid))
    Rc_true = integrate.quad(E, 0., z_lens)[0]
    Rc_fid = integrate.quad(E_fid, 0., z_lens)[0]
    rp_mid = (
        Rc_true / Rc_fid) * rp_mid_fid  # convert to true rp in this cosmology

    ds_true = np.zeros(rp_mid.shape[0])
    integrand = lambda r: r * wp(r, binmin, binmax, xi, pimax)
    for i in range(rp_mid.shape[0]):
        integral, abserr = integrate.quad(
            integrand, 0., rp_mid[i], epsabs=1.0e-3, epsrel=1.0e-3)
        ds_true[i] = (integral * (2.0 / rp_mid[i]**2) - wp(
            rp_mid[i], binmin, binmax, xi, pimax)) * mean_rho

    # convert Mpc/h unit to pc (no h)
    ds_true *= 1.0e6 / h
 
    Sigma_c_true = sigma_crit(z_lens, z_source,
    							Omega_m=Omega_m)	 # true Omega_m
    Sigma_c_assumed = sigma_crit(z_lens, z_source,
    							Omega_m=Omega_m_fid) # assumed Omega_m when making measurement
    
    gamma_t = ds_true / Sigma_c_true		# the observed gamma_t (in r_p units)
    ds_measured = gamma_t * Sigma_c_assumed	# the measured DeltaSigma

    return rp_binmin, rp_binmax, ds_measured


def compute_gamma_t(binmin,
                    binmax,
                    xi,
                    pimax=None,
                    H0=None,
                    Omega_m=None,
                    Omega_m_fid=0.3,
                    z_lens=None,
                    z_source=None,
                    rp_min=0.5,
                    rp_max=30.0,
                    nbins=30):
    binmin, binmax, DS = DeltaSigma(
        binmin,
        binmax,
        xi,
        pimax=pimax,
        H0=H0,
        Omega_m=Omega_m,
        Omega_m_fid=Omega_m_fid,
        z_lens=z_lens,
        rp_min=rp_min,
        rp_max=rp_max,
        nbins=nbins)
    Sigma_c = sigma_crit(z_lens, z_source, Omega_m=Omega_m)
    gamma_t = DS / Sigma_c
    return binmin, binmax, gamma_t


def DeltaSigma_from_files(header_file,
                          filename,
                          output_file,
                          pimax,
                          z_lens,
                          z_source,
                          rp_min,
                          rp_max,
                          nbins,
                          compute_DS=None):
                          
    ## read in cosmological parameters from header_file
    
    import config
    cf = config.AbacusConfigFile(header_file)
    omega_m = cf.Omega_M	# at z=0
    H_0 = 100.				# use h Msun pc^-3 units

    binmin, binmax, null, xi = np.loadtxt(filename, unpack=True)
    
    if compute_DS == False:
        DS_binmin, DS_binmax, DS = compute_gamma_t(
            binmin,
            binmax,
            xi,
            pimax=float(pimax),
            H0=H_0,
            Omega_m=omega_m,
            z_lens=z_lens,
            z_source=z_source,
            rp_min=float(rp_min),
            rp_max=float(rp_max),
            nbins=int(nbins))
    else:
        DS_binmin, DS_binmax, DS = DeltaSigma(
            binmin,
            binmax,
            xi,
            pimax=float(pimax),
            H0=H_0,
            Omega_m=omega_m,
            z_lens=z_lens,
            z_source=z_source,
            rp_min=rp_min,
            rp_max=rp_max,
            nbins=nbins)

    np.savetxt(
        output_file,
        np.c_[DS_binmin, DS_binmax,
              np.zeros(DS.shape[0]), DS],
        delimiter='\t')


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_file')
    parser.add_argument('header_file')
    parser.add_argument('output_file')
    
    parser.add_argument('--pimax', type=float, default=100.)
    parser.add_argument('--zlens', type=float, default=0.27)
    parser.add_argument('--zsource', type=float, default=0.447)

    parser.add_argument('--rpmin', type=float, default=0.5)
    parser.add_argument('--rpmax', type=float, default=30.0)
    parser.add_argument('--nbins', type=int, default=30)

    parser.add_argument('--DS', default=False, action='store_true')

    args = parser.parse_args()

    DeltaSigma_from_files(
        args.header_file,
        args.input_file,
        args.output_file,
        pimax=args.pimax,
        z_lens=args.zlens,
        z_source=args.zsource,
        rp_min=args.rpmin,
        rp_max=args.rpmax,
        nbins=args.nbins,
        compute_DS=args.DS)
