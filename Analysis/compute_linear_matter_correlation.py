""" python version of python FFTLOG.
    Algorithm due to Andrew Hamilton. 
    See http://casa.colorado.edu/~ajsh/FFTLog/

        Joseph E. McEwen
        email: jmcewen314@gmail.com

    Minor modifications of input/output formats and interpolation
    as part of this galaxy emulator pipline due to Ben Wibking.
    (June 2016) (ben@wibking.com).
"""

import numpy as np
from numpy.fft import fft, ifft , fftshift, ifftshift , rfft, irfft 
from numpy import exp, log, log10, cos, sin, pi
from scipy.special import gamma 
from time import time 
from numpy import gradient as grad
import sys

from compute_sigma8 import sigma_8_log_spaced, growth_factor

def log_gamma(z):
        z=gamma(z)
        w=log(z)
        x=np.real(w)
        y=np.imag(w)
        return x,y
        
def get_k0(N,mu,q,r0,L,k0):
        kr=float(k0*r0)
        delta_L=L/N
        
        x=q + 1j*pi/delta_L
        
        x_plus=(mu+1+x)/2.
        x_minus=(mu+1-x)/2.
                
        rp,phip=log_gamma(x_plus)
        rm,phim=log_gamma(x_minus)
        
        arg=log(2/kr)/delta_L + (phip - phim)/pi 
        iarg=np.rint(arg)
        if (arg != iarg):
                kr=kr*exp((arg-iarg)*delta_L)
                #kr=kr*exp((arg+iarg)*delta_L)          # Hamilton sign 
        return kr 

def u_m_vals(m,mu,q,kr,L):
        x=q + 1j*2*pi*m/L
        
        alpha_plus=(mu+1+x)/2.
        alpha_minus=(mu+1-x)/2.
                
        rp, phip=log_gamma(alpha_plus) 
        rm, phim=log_gamma(alpha_minus) 
        
        log2=log(2.)
        log_r=q*log2 + rp - rm 
        phi=2*pi*m/L*log(2./kr) + phip - phim 
        
        real_part=exp(log_r)*cos(phi)
        imag_part=exp(log_r)*sin(phi) 
        
        u_m=real_part + 1j*imag_part 
        
        # adjust endpoint, the N/2=m.size point 
        u_m[m.size-1]=np.real(u_m[m.size-1])
        return u_m

def fft_log(k,f_k,q,mu):
        """ Compute the discrete Fourier transform of a log-spaced input signal. """
        if ((q+mu) < -1):
                print('Error in reality condition for Bessel function integration.')
                print(' q+mu is less than -1.')
                print('See Abramowitz and Stegun. Handbook of Mathematical Functions pg. 486')
                sys.exit(1)
        if ( q > 1/2.):
                print('Error in reality condition for Bessel function integration.')
                print(' q is greater than 1/2')
                print('See Abramowitz and Stegun. Handbook of Mathematical Functions pg. 486')
                sys.exit(1)
                                                
        N=f_k.size
        delta_L=(log(np.max(k))-log(np.min(k)))/(N-1)
        #delta_L10=(np.log10(np.max(k))-np.log10(np.min(k)))/(N-1)
        L=(log(np.max(k))-log(np.min(k)))
                
        log_k0=log(k[N//2])
        k0=exp(log_k0)
        
        # Fourier transform input data 
        # get m values, shifted so the zero point is at the center
        
        c_m=rfft(f_k)
        m=np.fft.rfftfreq(N,d=1.)*float(N)
        # make r vector 
        #kr=get_k0(float(N),mu,q,1/k0,L,k0)
        kr=1
        r0=kr/k0
        log_r0=log(r0)
        
        m=np.fft.rfftfreq(N,d=1.)*float(N)
        m_r=np.arange(-N//2,N//2) # this must be an array of ints to work in py3
        m_shift=np.fft.fftshift(m_r)
        
        #s-array 
        s=delta_L*(-m_r)+log_r0         
        id=m_shift
        r=10**(s[id]/log(10))
        
        #m_shift=np.fft.fftshift(m)
        
        # get h array   
        h=delta_L*m + log_k0
                
        u_m=u_m_vals(m,mu,q,kr,L)

        b=c_m*u_m
                
        A_m=irfft(b)
        
        A=A_m[id]
        
        # reverse the order 
        A=A[::-1]
        r=r[::-1]
        
        if (q!=0):
                A=A*(r)**(-float(q))
                
        return r, A 

##########################################################################################
# End of fftlog algorithm 


def k_to_r(k, f_k, alpha_k=1.5, beta_r=-1.5, mu=.5, pf=(2*pi)**(-1.5), q=0):
        """
        module to calculate Hankel Transform
        \int_0^\infty dk r A(k) J_mu(kr), via fftlog algorithm
        Common application is for power spectrum:
        \\xi(r)= \int dk k^2 /(2 \pi^2) \sin(kr)/kr P(k) 
        in which case 
        alpha_k=1.5
        beta_r=-1.5
        mu=.5 
        pf=(2*np.pi)**(-1.5)
        """
        
        f_k=k**alpha_k*f_k
        r, A=fft_log(k,f_k,q,mu)
        f_r=pf*A*r**beta_r 
        
        return r, f_r 
        
def r_to_k(r, f_r, alpha_k=-1.5, beta_r=1.5, mu=.5, pf=4*pi*np.sqrt(pi/2.), q=0):
        """
        module to calculate Hankel Transform
        \int_0^\infty dr k A(r) J_mu(kr), via fftlog algorithm
        Common application is for correlation function:
        P(k)= 2 pi \int dr r^2  \sin(kr)/kr xi(r) 
        in which case 
        alpha_k=-1.5
        beta_r=1.5
        mu=.5 
        pf=4 pi *sqrt(pi/2)
        """
        
        f_r=r**beta_r*f_r
        k, A=fft_log(r,f_r,q,mu)
        f_k=pf*A*k**alpha_k 

        return k, f_k


""" Compute linear correlation function from input power spectrum """
if __name__=='__main__':
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('input_file')
        parser.add_argument('min_r_bin',type=float)
        parser.add_argument('max_r_bin',type=float)
        parser.add_argument('nbins',type=int)
        parser.add_argument('header_file')
        parser.add_argument('output_file')
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

        # read in power spectrum
        data=np.loadtxt(args.input_file)
        k_camb=data[:,0]
        P_camb=data[:,1]
        log_k_camb = np.log10(k_camb)
        
        # convert to uniform sampling in log-k (CAMB output is *not* log-spaced!)
        from scipy.interpolate import interp1d
        P_interp = interp1d(log_k_camb, P_camb)
        
        # extrapolate past k_camb.max()
        logkmax = np.log10(k_camb[-1])
        Pmax = P_camb[-1]
        P_asymp = lambda logk: Pmax * (10.**(-3.0*(logk-logkmax)))
        
        def P_interp_and_asymp(logk):
                if logk >= logkmax:
                        return P_asymp(logk)
                else:
                        return P_interp(logk)

        P_vec = np.vectorize(P_interp_and_asymp)
                        
        k = np.logspace(log_k_camb.min(), 4.0, 2048)
        log_k = np.log10(k)
        dlogk = np.log(k[1]/k[0]) # natural log here!
        P = P_vec(log_k)

        # set the sigma_8 normalization to the input sigma_8
        # (separately, adjust the normalization by the growth factor at a given redshift)
        input_sigma_8 = sigma_8_log_spaced(P,k=k)
        this_growth_factor = growth_factor(redshift=redshift, omega_m=omega_m)
        rescale_factor = (this_growth_factor * target_sigma_8 / input_sigma_8)**2

        P *= rescale_factor

        # compute correlation function via FFTLOG
        r,xi=k_to_r(k,P)
        k2,P2=r_to_k(r,xi)

        # resample correlation function to log-spaced bins from 0.1 to 30 Mpc/h
        xi_interp = interp1d(r,xi)
        bin_edges = np.logspace(np.log10(args.min_r_bin), np.log10(args.max_r_bin), args.nbins+1)
        binmin = bin_edges[0:-1]
        binmax = bin_edges[1:]

        # loop over bins, compute volume-average integrals
        from scipy.integrate import quad
        xi_resampled = np.zeros(binmin.shape[0])
        for i in range(binmin.shape[0]):
                bin_integrand = lambda r: xi_interp(r) * r**2
                bin_integral, abserr = quad(bin_integrand, binmin[i], binmax[i], epsabs=1.0e-4, epsrel=1.0e-4)
                bin_integral *= 3.0/(binmax[i]**3 - binmin[i]**3)
                xi_resampled[i] = bin_integral

        np.savetxt(args.output_file, np.c_[binmin, binmax, np.zeros(xi_resampled.shape[0]), xi_resampled], delimiter='\t')

