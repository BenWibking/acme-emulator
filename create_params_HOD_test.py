import numpy as np
import configparser

#from fisher_Rockstar import use_log_parameter
use_log_parameter = ['ngal','ncen','siglogM','M1_over_Mmin','alpha','H0','sigma_8','M0_over_M1','Omega_M']#,'f_cen']
cosmological_parameters = ['H0', 'sigma_8', 'Omega_M']

def read_cov_file(filename):
    """Read parameter covariance file"""
    cov = np.genfromtxt(filename, names=True)
    param_names = list(cov.dtype.names)
    cov_arr = cov.view(np.float64).reshape(cov.shape[0],cov.shape[0])
    return cov_arr, param_names

def write_param_file(base_filename, fiducial_config, new_params, file_id, parameter=None):
    """write out parameter file with parameters 'params'.
    [1. read in fiducial parameter file]
    2. change varied parameters
    3. write out new file
    """
    fiducial_config['params']['parameter'] = parameter

    for param,value in new_params:
        if param not in cosmological_parameters: # don't change cosmological parameters
            fiducial_config['params'][param] = str(value)

    output_filename = "{}.{}.template_param".format(base_filename, file_id)
    print("writing to {}".format(output_filename))
    with open(output_filename,'w') as output_file:
        fiducial_config.write(output_file)

def main(args):
    """write out parameter files for random samples from the inverse Fisher matrix"""
    cov, param_names = read_cov_file(args.parameter_cov_filename)
    sqrt_diag_cov = np.sqrt(np.diag(cov))
    nsamples = int(args.number)

    # read fiducial param file
    fiducial_config = configparser.ConfigParser()
    fiducial_config.read(args.fiducial_param_file)
    fiducial_params = np.zeros(len(param_names))
    param_uncertainty = np.zeros(len(param_names))
    for i, param_name in enumerate(param_names):
        fiducial_params[i] = fiducial_config['params'][param_name]
        if param_name in use_log_parameter: # we have a fractional uncertainty
            param_uncertainty[i] = sqrt_diag_cov[i] * fiducial_params[i] # convert to absolute uncertainty
        else:
            param_uncertainty[i] = sqrt_diag_cov[i]

    hod_param_index = []
    for i, param in enumerate(param_names):
        if param not in cosmological_parameters:
            hod_param_index.append(i)

    rescale_sigmas = float(args.rescale_sigmas)

    corr = np.zeros(cov.shape)
    for k in range(corr.shape[0]):
        for l in range(corr.shape[1]):
            corr[k,l] = cov[k,l] / np.sqrt(cov[k,k]*cov[l,l])

    RandomGenerator = np.random.RandomState(seed=42) # make reproducible randomness
    for i in range(nsamples):
         # uniform random sample -3\sigma to +3\sigma changes in individual parameters
        this_parameter = 'None'
        if args.single_parameter == True:
            # only vary one HOD parameter at a time
            sigmas = np.zeros(sqrt_diag_cov.shape[0])
            idx = hod_param_index[RandomGenerator.randint(len(hod_param_index))]
            sigmas[idx] = 6.0 * (RandomGenerator.rand() - 0.5)
            this_parameter = param_names[idx]
        elif args.sample_posterior == True:
            # sample from the multivariate Gaussian defined by 'corr'
            sigmas = rescale_sigmas * RandomGenerator.multivariate_normal(np.zeros(corr.shape[0]), corr)
            # print out chi^2 for this sample
            dof = sigmas.shape[0]
            chisq_dof = np.dot(sigmas, np.dot(corr, sigmas)) / dof
            print('chisq/dof = {}'.format(chisq_dof))
        else:
            sigmas = 6.0 * (RandomGenerator.rand(sqrt_diag_cov.shape[0]) - 0.5)
        delta_params = param_uncertainty * sigmas
        new_param_values = fiducial_params + delta_params
        new_params = zip(param_names, new_param_values)
        write_param_file(args.output_filename_base, fiducial_config, new_params, i,
                         parameter=this_parameter)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fiducial_param_file')
    parser.add_argument('parameter_cov_filename')
    parser.add_argument('output_filename_base')
    parser.add_argument('--number',type=int,default=1)
    parser.add_argument('--single-parameter',default=False,action='store_true')
    parser.add_argument('--sample-posterior',default=False,action='store_true')
    parser.add_argument('--rescale-sigmas',type=float,default=1.0)
    args = parser.parse_args()

    main(args)
