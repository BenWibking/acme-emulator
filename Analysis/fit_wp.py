import numpy as np
import scipy.optimize
import argparse
from emulator import emulator_data_from_files, emulate
from compute_wp import wp

def read_observable(filename):
    table = np.loadtxt(filename,unpack=False)
    binmin, binmax, zeros, obs = [table[:,i] for i in range(4)]
    return binmin,binmax,obs

def fit_files(cov_file, obs_file, xi_mm_file, fiducial_file, deriv_files,
              output_file_model, output_file_params, Omega_m=0.3, z_lens=0.27):
    # read inverse covariance matrix
    cov = np.loadtxt(cov_file)
    inv_cov = np.linalg.inv(cov)

    # read observable (wp)
    binmin_wp,binmax_wp,observable = read_observable(obs_file)
    nobs = len(observable)
    # ensure that dimensions match
    if(observable.shape[0] == inv_cov.shape[0] == inv_cov.shape[1]):
        pass
    else:
        print("observable:",observable.shape[0], "covariance:", inv_cov.shape[0], inv_cov.shape[1])
        raise Exception("dimensions of observable and covariance matrix do not match!")

    # read emulator data for ln b_g
    deriv_filenames, fiducial_params, param_min, param_max = zip(*deriv_files)
    param_bounds = list(zip(param_min, param_max))
    print(param_bounds)
    fiducial_params = np.array([float(x) for x in fiducial_params])
    data = emulator_data_from_files(fiducial_file, deriv_filenames)
    binmin_ln_bg, binmax_ln_bg, fiducial_ln_bg, derivs_ln_bg = data
    nparams = len(fiducial_params)

    # read xi_mm
    binmin_ximm, binmax_ximm, null, xi_mm = np.loadtxt(xi_mm_file, unpack=True)
    # ensure that bins match
    if(np.allclose(binmin_ln_bg,binmin_ximm) and np.allclose(binmax_ln_bg,binmax_ximm)):
        pass
    else:
        raise Exception("bins of xi_mm and ln_bg do not match!")

    # set up emulator for wp
    ln_bg = lambda params: emulate(fiducial_ln_bg, derivs_ln_bg, params, fiducial_params)
    bg_sq = lambda params: np.exp(2.0*ln_bg(params))
    xi_gg = lambda params: bg_sq(params) * xi_mm
    wp_model = lambda params: wp(binmin_ximm, binmax_ximm, xi_gg(params), Omega_m, z_lens, pimax=100.)[2]

    # do fit here, varying params
    model = lambda params: wp_model(params)
    delta_p = lambda params: (model(params) - observable)
    chi_sq = lambda params: np.dot(delta_p(params), np.dot(inv_cov, delta_p(params)))

    x0 = fiducial_params
    result = scipy.optimize.minimize(chi_sq, x0, method='L-BFGS-B', bounds=param_bounds)
    opt_params = result.x
    opt_model = model(opt_params)
    opt_chi_sq = chi_sq(opt_params)
    dof = nobs - nparams
    opt_reduced_chi_sq = opt_chi_sq / dof
    print("optimized chi^2:",opt_chi_sq,"(reduced:",opt_reduced_chi_sq,")")
    print("optimized parameters:",opt_params)
    print("optimized exp parameters:",np.exp(opt_params))

    # save optimized model
    np.savetxt(output_file_model, np.c_[binmin_wp, binmax_wp, np.zeros(opt_model.shape[0]), opt_model],
		   delimiter='\t')
    np.savetxt(output_file_params, opt_params)

parser = argparse.ArgumentParser()
parser.add_argument('output_wp',help='txt file output for emulated observable (wp)')
parser.add_argument('output_params',help='txt file output for best-fit parameters')
parser.add_argument('wp_obs_file')
parser.add_argument('wp_cov_file')
parser.add_argument('xi_mm_file') # assume fixed cosmology for now
parser.add_argument('fiducial_ln_bg',help='txt file output for fiducial emulated observable')
parser.add_argument('-f','--deriv_ln_bg',nargs=4,action='append',help='derivative file')
# this returns a list of tuples, one item for each input file
# -- the first part of the tuple is the filename
# -- the second part of the tuple is the parameter value of the fiducial model
# -- the third part of the tuple is the minimum allowed parameter value when fitting
# -- the fourth part of the tuple is the maximum allowed parameter value when fitting

args = parser.parse_args()

fit_files(args.wp_cov_file, args.wp_obs_file, args.xi_mm_file, args.fiducial_ln_bg, args.deriv_ln_bg,
          args.output_wp, args.output_params)
