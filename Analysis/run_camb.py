import numpy as np
from numpy import exp, log, log10, cos, sin, pi

import camb
import camb.model

def linear_pk(omega_m, omch2=0.1199, ombh2=0.02222, w0=-1.0, ns=0.9652, kmin=1e-4, kmax=1e2,
              redshift=0., do_nonlinear=False):
        """return (non)linear power spectrum"""
        H_0 = np.sqrt((omch2+ombh2) / omega_m) * 100.

        camb_params = camb.CAMBparams()
        camb_params.set_cosmology(H0=H_0, ombh2=ombh2, omch2=omch2, mnu=0)
        camb_params.set_dark_energy(w=w0)
        camb_params.InitPower.set_params(ns=ns, As=2.1e-9)
        camb_params.set_matter_power(redshifts=[redshift], kmax=kmax)
        camb_params.set_accuracy(AccuracyBoost=3, lAccuracyBoost=3)
        if do_nonlinear == True:
                camb_params.NonLinear = camb.model.NonLinear_pk
        else:
                camb_params.NonLinear = camb.model.NonLinear_none

        results = camb.get_results(camb_params)
        k, z, pk = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints=3000)
        P = pk[0,:]
        return k, P

if __name__=='__main__':
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('header_file')
        parser.add_argument('output_file')
        parser.add_argument('--halofit',default=False,action='store_true')
        parser.add_argument('--redshift',type=float,default=0.)
        args = parser.parse_args()

        # read in cosmological parameters from header_file
        import config
        cf = config.AbacusConfigFile(args.header_file)
        omega_m = cf.Omega_M # at z=0
        redshift = cf.redshift
        target_sigma_8 = cf.sigma_8
        H_0 = cf.H0
        omch2 = cf.omch2
        ombh2 = cf.ombh2
        w0 = cf.w0
        ns = cf.ns

        k, P = linear_pk(omega_m, omch2=omch2, ombh2=ombh2, w0=w0, ns=ns, redshift=args.redshift,
                         do_nonlinear=args.halofit)
        np.savetxt(args.output_file, np.c_[k, P], delimiter='\t')
