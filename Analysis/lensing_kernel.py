import numpy as np
import scipy.integrate
import scipy.optimize
import matplotlib.pyplot as plt


def D_C(z, Omega_M):
	"""compute comoving radial distance to redshift z assuming flat LCDM."""
	
	H = lambda z: 1.0/np.sqrt( Omega_M*(1.0 + z)**3 + (1.0 - Omega_M) )
	chi, abserr = scipy.integrate.quad(H, 0., z)

	return 3000.0 * chi

def redshift(chi, Omega_M):
	"""compute redshift as a function of comoving distance."""
	
	fun = lambda z: D_C(z, Omega_M) - chi
	z_guess = 1.0
	z = scipy.optimize.newton(fun, z_guess)

	return z


def lensing_window_function(Pi, z_lens, z_source, Omega_M=0.3):
	"""compute the lensing window function W(\Pi) assuming flat LCDM."""

	chi_lens = D_C(z_lens, Omega_M=Omega_M)
	chi_source = D_C(z_source, Omega_M=Omega_M)
	chi_lens_plus_pi = chi_lens + Pi

	z_lens_plus_pi = redshift(chi_lens_plus_pi, Omega_M=Omega_M)

	return ((1.0 + z_lens_plus_pi)/(1.0 + z_lens)) * \
			( chi_lens_plus_pi * (chi_source - chi_lens_plus_pi) ) / ( chi_lens * (chi_source - chi_lens) )


if __name__ == '__main__':
	"""plot the lensing window function."""

	"""replace these redshift with those of your survey."""
	z_lens = 0.5
	z_source = 1.0

	chi_lens = D_C(z_lens, Omega_M=0.3)
	chi_source = D_C(z_source, Omega_M=0.3)
	min_pi = -chi_lens
	max_pi = chi_source - chi_lens

	lensing_window = lambda chi: lensing_window_function(chi, z_lens, z_source)
	W = np.vectorize(lensing_window)
	chi = np.linspace(min_pi, max_pi, 1e3)

	delta_Pi1, abserr = scipy.integrate.quad(W, min_pi, max_pi)
	print("Delta_Pi_1 = {}".format(delta_Pi1))

	""" Delta_Pi2 (below) is the squared lensing kernel width that goes into the covariance formula.
		(I'm not sure whether Delta_Pi1 or Delta_Pi2 should go into the integration limits for DeltaSigma...)"""

	Wsq = lambda chi: lensing_window(chi)**2
	delta_Pi2, abserr = scipy.integrate.quad(Wsq, min_pi, max_pi)
	print("Delta_Pi_2 = {}".format(delta_Pi2))


	""" Plot the lensing kernel."""

	plt.figure()
	plt.plot(chi, W(chi), label='lensing window function')
	plt.plot(chi, W(chi)**2, label='lensing squared window')
	plt.xlabel(r'distance offset from lens redshift')
	plt.legend(loc='best')
	plt.tight_layout()
	plt.savefig('lensing_kernel.pdf')