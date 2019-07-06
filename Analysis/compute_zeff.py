import numpy as np
import scipy.integrate
import argparse


def comoving(redshift, Omega_m=0.3):

	"""compute comoving radial distance to redshift 'redshift' in flat LCDM."""

	norm_fac = 3000.0 # Mpc/h
	E = lambda z: Omega_m*(1+z)**3 + (1-Omega_m)	# ignore radiation, neutrinos, etc.
	integrand = lambda z: 1.0/np.sqrt(E(z))
	chi = norm_fac * scipy.integrate.quad(integrand, 0., redshift)[0]	# Mpc/h units

	return chi


def sigma_crit(z_lens, z_source, Omega_m=0.3):

	D_H = 3000.0  # Mpc/h
	D_H_pc = D_H * 1.0e6  # pc/h
	three_csq_over_8piG = 2.494e12  # 3c^2/(8*pi*G) Msun pc^-1
	csq_over_4piG = three_csq_over_8piG * (2. / 3.)
	Omega_L = 1.0 - Omega_m

	integrand = lambda z: 1.0 / np.sqrt(Omega_m * (1.0 + z)**3 + Omega_L)
	D_lens = D_H_pc * scipy.integrate.quad(integrand, 0., z_lens)[0]
	D_source = D_H_pc * scipy.integrate.quad(integrand, 0., z_source)[0]

	dist_fac = D_source / ((1.0 + z_lens) * (D_lens * (D_source - D_lens)))
	sigma_c = csq_over_4piG * dist_fac  # h Msun pc^-2
	return sigma_c


if __name__ == '__main__':

	"""compute the effective redshift for clustering (or lensing)
	given an input dn/dz (and dn_src/dz)."""

	parser = argparse.ArgumentParser()
	parser.add_argument('input_dndz')
	parser.add_argument('--sources')
	args = parser.parse_args()

	vec_comoving = np.vectorize(comoving)
	vec_sigma_crit = np.vectorize(sigma_crit)

	omegam = 0.3


	## read in dn/dz (lenses)

	zcen, zlow, zhigh, nbar, wfkp, shell_vol, Ngals = np.genfromtxt(args.input_dndz, unpack=True)

	## restrict to 0.16 < z < 0.36
	zmin = 0.16
	zmax = 0.36
	zmask = np.logical_and( zlow > zmin, zhigh < zmax )
	print(zcen[zmask])
	Ngals[~zmask] = 0.

	dz = zhigh - zlow
	dndz = Ngals / dz
	chi = vec_comoving(zcen)
	dchidz = 3000.0 / np.sqrt( omegam*(1+zcen)**3 + (1-omegam) )
	dV_dz = chi**2 * dchidz
	
	clustering_weights = (dndz**2 / dV_dz)
#	zeff = np.sum(clustering_weights * zcen * dV_dz * dz) / np.sum(clustering_weights * dV_dz * dz)
	zeff = np.sum(clustering_weights * zcen * dz) / np.sum(clustering_weights * dz)

	print(f"zeff_clustering = {zeff}")

	
	## read in dn/dz (sources)

	zspec, ra, dec, zphot = np.genfromtxt(args.sources, unpack=True)

	Ns_counts, bin_edges = np.histogram(zspec, bins=100, density=False)
	dz_s = bin_edges[1:] - bin_edges[:-1]
	zs_cen = 0.5*(bin_edges[1:] + bin_edges[:-1])
	dns_dz = Ns_counts / dz_s

	def w_l(z_l):
		"""compute lensing weights."""

		## compute luminosity distances to z_l
		D_L = (1 + z_l) * comoving(z_l)

		## compute Sigma_c integrated over dn_src/dz
		integrand = np.zeros_like(zs_cen)
		mask = tuple([zs_cen > z_l])	# no lensing when z_s <= z_l

		integrand[mask] = dns_dz[mask] * vec_sigma_crit(z_l, zs_cen[mask])**(-2)
		integral = np.sum(integrand * dz_s)

		return D_L**(-2) * (1.0 + z_l)**(-2) * integral


	vec_wl = np.vectorize(w_l)
	lensing_weights = dndz * vec_wl(zcen)

#	zeff_lensing = np.sum(lensing_weights * zcen * dV_dz * dz) / np.sum(lensing_weights * dV_dz * dz)
	zeff_lensing = np.sum(lensing_weights * zcen * dz) / np.sum(lensing_weights * dz)

	print(f"zeff_lensing = {zeff_lensing}")
