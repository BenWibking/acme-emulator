import numpy as np
from scipy.integrate import quad
import sys

def growth_factor(redshift, omega_m):
        # compute linear growth factor
        H = lambda z: np.sqrt(omega_m * (1.0+z)**(3.0) + (1.0-omega_m)) # assume flat LCDM
        integrand = lambda z: (1.0+z) * H(z)**(-3.0)
        norm, abserr = quad(integrand, 0., np.inf)
        growth_factor, abserr = quad(integrand, redshift, np.inf)
        growth_factor *= H(redshift) / norm
        return growth_factor

""" Rescale input power spectrum by growth factor"""
if __name__=='__main__':
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('input_file',help='z=input_redshift input correlation function')
        parser.add_argument('input_redshift',type=float)
        parser.add_argument('header_file')
        parser.add_argument('output_file')
        args = parser.parse_args()

        input_redshift = args.input_redshift

        # read in cosmological parameters from header_file
        import config
        cf = config.AbacusConfigFile(args.header_file)
        omega_m = cf.Omega_M # at z=0
        output_redshift = cf.redshift
        H_0 = cf.H0

        print('output_redshift z=%s' % output_redshift)
        print('input_redshift z=%s' % input_redshift)

        # read in power spectrum
        data=np.loadtxt(args.input_file)
        binmin=data[:,0]
        binmax=data[:,1]
        corr=data[:,3]

        output_growth_factor = growth_factor(output_redshift, omega_m)
        input_growth_factor = growth_factor(input_redshift, omega_m)
        rescale_factor = (output_growth_factor/input_growth_factor)**2

        corr_rescaled = corr * rescale_factor

        print('output D^2: %s' % output_growth_factor**2)
        print('input D^2: %s' % input_growth_factor**2)
        print('rescale factor: %s' % rescale_factor)

        np.savetxt(args.output_file, np.c_[binmin, binmax, np.zeros(binmin.shape[0]), corr_rescaled], delimiter='\t')

