import numpy as np
import scipy.interpolate

def wcdm_growth_factor(redshift=None, omega_m=None, w0=-1.0, wa=0.):
	"""compute wCDM growth factor with (w0, wa) parameterization.
		omega_m here refers to omega_m(z=0)."""

	f = lambda a: a**(-3.0*(w0+wa)) * np.exp(3.0*wa*(a-1.0))
	omega_de = 1.0 - omega_m

	E_sq = lambda a: omega_m*a**(-3) + omega_de*f(a)*a**(-3)
	O_m = lambda a: omega_m*a**(-3) / E_sq(a)
	dlnHsq_dlna = lambda a: (-3.0*a**(-3)/E_sq(a)) * (omega_m + omega_de*f(a)*(w0 + wa*(1.0-a) + 1.0))
	
	def ode_system(lna, y):
		"""define ODEs for growth factor."""
		g  = y[0]	# == g[ln a]
		gp = y[1]	# == g'[ln a]

		a = np.exp(lna)
		yp = np.empty(y.shape)
		yp[0] = gp
		yp[1] = -(4.0 + 0.5*dlnHsq_dlna(a)) * gp - (3.0 + 0.5*dlnHsq_dlna(a) - 1.5*O_m(a)) * g

		return yp

	from scipy.integrate import solve_ivp
	ln_a_i = -10.
	ln_a_f = 0.
	y0 = [1.0, 0.0]
	solution = solve_ivp(ode_system, (ln_a_i, ln_a_f), y0, method='RK45', atol=1e-5)
	lna_output = solution.t
	g		 = solution.y[0,:]
	dg_dlna	 = solution.y[1,:]

	## normalize s.t. D(z=0) = 1.
	a_output = np.exp(lna_output)
	D_a  = g*a_output
	D_a *= (1.0/D_a[-1])

	D = scipy.interpolate.interp1d(a_output, D_a)
	input_a = 1.0/(1.0+redshift)
	growth = D(input_a)

	return growth


def wcdm_growth_factor_cpt(redshift=None, omega_m=None, w0=-1.0, wa=0.):
	"""compute Carroll, Press & Turner (1992) fitting formula."""

	f = lambda a: a**(-3.0*(w0+wa)) * np.exp(3.0*wa*(a-1.0))
	omega_de = 1.0 - omega_m

	E = lambda a: np.sqrt( omega_m*a**(-3) + omega_de*f(a)*a**(-3) )
	O_m = lambda a: omega_m*a**(-3) / E(a)**2
	O_de = lambda a: omega_de*f(a)*a**(-3) / E(a)**2
	g_cpt = lambda a: (5./2.)*O_m(a) * ( O_m(a)**(4./7.) - O_de(a) \
						+ ( 1.0 + (O_m(a) / 2.0) ) * ( 1.0 + O_de(a) / 70. ) )**(-1)

	input_a = 1.0/(1.0+redshift)
	growth_cpt = (input_a * g_cpt(input_a)) / (1.0 * g_cpt(1.0))

	return growth_cpt


if __name__ == "__main__":
	"""compute tests."""

	from compute_sigma8 import growth_factor

	def growth(redshift, om=0.3, w0=-1., wa=0.):
		D = wcdm_growth_factor(redshift=redshift, omega_m=om, w0=w0, wa=wa)
		D_cpt = wcdm_growth_factor_cpt(redshift=redshift, omega_m=om, w0=w0, wa=wa)
		D_lcdm = growth_factor(redshift=redshift, omega_m=om)
		print("growth factor D(z={}) = {} (cpt: {}, LCDM: {})".format(redshift,D,D_cpt,D_lcdm))

	growth(0.)
	growth(0.5)
	growth(1.0)