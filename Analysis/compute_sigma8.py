import numpy as np
from numpy import exp, log, log10, cos, sin, pi
from scipy.integrate import quad
from scipy.integrate import simps as simpson
from scipy.integrate import odeint
from scipy.interpolate import interp1d


def camb_linear_pk(H_0=None,ombh2=None,omch2=None,w0=None,redshift=0.):
	import camb
	camb_params = camb.CAMBparams()
	camb_params.set_cosmology(H0=H_0, ombh2=ombh2, omch2=omch2, mnu=0)
	camb_params.set_dark_energy(w=w0)
	camb_params.InitPower.set_params(ns=ns, As=2.1e-9)
	camb_params.set_matter_power(redshifts=[redshift], kmax=32.)
	camb_params.NonLinear = camb.model.NonLinear_none
	results = camb.get_results(camb_params)
	k, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1e4,npoints=2048)
	camb_sigma_8 = results.get_sigma8()[0]
	P = pk[0,:]
	return k,P,camb_sigma_8


def wcdm_growth_factor(redshift, omega_m=None, w0=None, wa=0.):

	"""compute linear growth factor in a wCDM universe. (code borrowed from Hao-Yi Wu.)"""

	omega_de = 1.0 - omega_m	# assume zero curvature

	def OmegaM(ln_a):
		return omega_m*exp( -3.0*ln_a ) / Esq(ln_a)

	def Esq(ln_a):
		a = exp(ln_a)
		return omega_m * exp( -3.0*ln_a ) + \
			   omega_de * exp( -3.0*(1.0+w0+wa)*ln_a + 3.0*wa*(a-1.0) )

	def dlnHsq(ln_a):
		eps = 1e-4
		return ( log( Esq(ln_a) ) - log( Esq(ln_a-eps) ) ) / eps

	def f(y, ln_a):
		dfdt = -( 4.0 + 0.5*dlnHsq(ln_a) ) * y[1] \
				- ( 3.0 + 0.5*dlnHsq(ln_a) - 1.5 * OmegaM(ln_a) ) * y[0]
		dydt = [y[1], dfdt]
		return dydt

	ln_a = np.linspace(-20, 0, 200)
	y0 = [1., 0.]
	result = odeint(f, y0, ln_a)
	g = result[:,0]
	a = np.exp(ln_a)
	z = 1.0/a - 1.0
	D_growth = g*a
	D_growth /= np.max(D_growth)

	interp_D_growth = interp1d(z, D_growth)
	return interp_D_growth(redshift)


def growth_factor(redshift=None,omega_m=None):

	"""	compute linear growth factor. """

	## assuming flat LCDM [from David's notes]
	H = lambda z: np.sqrt(omega_m * (1.0+z)**(3.0) + (1.0-omega_m))
	integrand = lambda z: (1.0+z) * H(z)**(-3.0)
	this_growth_factor, abserr = quad(integrand, redshift, np.inf)
	norm, abserr = quad(integrand, 0., np.inf)
	this_growth_factor *= H(redshift) / (H(0.) * norm)

	## from Hamilton 2000
#	H = lambda a: np.sqrt(omega_m * a**(-3.0) + (1.0 - omega_m))
#	a = 1.0/(1.0+redshift)
#	integrand = lambda ap: (ap*H(ap))**(-3.0)
#	g, abserr_g = quad(integrand, 0., a)
#	g0, abserr_g0 = quad(integrand, 0., 1.)
#	D = (H(a) * g) / (H(1.0) * g0)
#	print("{} ?= {}".format(this_growth_factor, D))

	return this_growth_factor


def sigma_8_log_spaced(P,k=None):
	R = 8.0 # Mpc h^-1
	def W(k,r):
			return 3.0*(np.sin(k*R) - k*R*np.cos(k*R)) / (k*R)**3
	dlogk = np.log(k[1]/k[0]) # natural log here! (input Pk must be log-spaced!)        
	input_sigma_8 = np.sqrt(simpson(P * k**3 * W(k,R)**2 / (2 * np.pi**2), dx=dlogk))
	return input_sigma_8


""" Compute linear correlation function from input power spectrum """
if __name__=='__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--input_file',default=None)
	parser.add_argument('header_file')
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
	if args.input_file is not None:
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
		P = P_vec(log_k)

	else:
		k,P,camb_sigma_8 = camb_linear_pk(H_0=H_0,omch2=omch2,ombh2=ombh2,w0=w0,
									redshift=0.)
		kz,Pz,camb_sigma_8_z = camb_linear_pk(H_0=H_0,omch2=omch2,ombh2=ombh2,w0=w0,
										redshift=redshift)
		log_k = np.log10(k)
		dlogk = np.log(k[1]/k[0]) # natural log here!


	# set the sigma_8 normalization to the input sigma_8
	# (separately, adjust the normalization by the growth factor at a given redshift)

	input_sigma_8 = sigma_8_log_spaced(P,k=k)
	this_growth_factor = growth_factor(redshift=redshift,omega_m=omega_m)
	#rescale_factor = (this_growth_factor * target_sigma_8 / input_sigma_8)**2
	rescale_factor = (this_growth_factor)**2

	P *= rescale_factor

	sigma_8_z = this_growth_factor * input_sigma_8

	print('sigma_8(z): %s' % sigma_8_z)
	print('CAMB sigma_8(z): %s' % camb_sigma_8_z)

	print('sigma_8(z=0): %s' % input_sigma_8)
	print('CAMB sigma_8(z=0): %s' % camb_sigma_8)

	print(P/Pz)
	print(np.mean(P/Pz))

	print('target sigma_8: %s' % target_sigma_8)
	#print('rescale_factor: %s' % rescale_factor)

	#print('D^2: %s' % growth_factor**2)

