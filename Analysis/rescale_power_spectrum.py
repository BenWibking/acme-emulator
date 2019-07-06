import numpy as np
from numpy import exp, log, log10, cos, sin, pi
import sys

""" Rescale input power spectrum by growth factor"""
if __name__=='__main__':
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('input_file',help='z=0 input power spectrum')
        parser.add_argument('header_file')
        parser.add_argument('output_file')
        args = parser.parse_args()

        # read in cosmological parameters from header_file
        import config
        cf = config.AbacusConfigFile(args.header_file)
        omega_m = cf.Omega_M # at z=0
        redshift = cf.redshift
        H_0 = cf.H0

        # read in power spectrum
        data=np.loadtxt(args.input_file)
        k_camb=data[:,0]
        P_camb=data[:,1]

        # compute linear growth factor
        from scipy.integrate import quad
        H = lambda z: np.sqrt(omega_m * (1.0+z)**(3.0) + (1.0-omega_m)) # assume flat LCDM
        integrand = lambda z: (1.0+z) * H(z)**(-3.0)
        growth_factor, abserr = quad(integrand, redshift, np.inf)
        norm, abserr = quad(integrand, 0., np.inf)
        growth_factor *= H(redshift) / norm

        P_rescaled = P_camb * (growth_factor**2)

        print('Omega_m: %s' % omega_m)
        print('D^2: %s' % growth_factor**2)

        np.savetxt(args.output_file, np.c_[k_camb, P_rescaled], delimiter='\t')

