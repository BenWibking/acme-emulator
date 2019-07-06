from pipeline_defs import *
from pathlib import Path
import shlex
import configparser
from itertools import chain

"""
N.B.: For some reason doit runs tasks whose 'file_dep's do not exist. This is not what we want...
Workaround: Use 'task_dep' to additionally specify the tasks that create the 'file_dep's.
"""

projected_rmin = "0.1"
projected_rmax = "30.0"
projected_nbins = "35"

all_varied_parameters = ['ngal','siglogM','M1_over_Mmin','alpha','q_env','del_gamma','Omega_M','sigma_8']#,'f_cen']
all_varied_parameters_centralsonly = ['ncen','siglogM','q_env','Omega_M','sigma_8'] # *don't* use ngal here

use_log_parameter = ['ngal','ncen','siglogM','M1_over_Mmin','alpha','H0','sigma_8','M0_over_M1','Omega_M']#,'f_cen']
cosmological_parameters = ['H0', 'sigma_8', 'Omega_M']
param_dir_alias = {}
param_dir_alias['Omega_M'] = 'H0'
param_dir_alias['sigma_8'] = 'sigma_8'
param_dir_alias['H0'] = 'H0'

all_param_files = param_files_in_dir(param_dir)
all_param_files_centralsonly = param_files_in_dir_centralsonly(param_dir)

verbose=False
def param_files_with_parameter_variation(files, parameter=None):
    result = []
    for file in files:
        myconfigparser = configparser.ConfigParser()
        myconfigparser.read(str(file))
        params = myconfigparser['params']
        if verbose:
            print('{} {}'.format(params['parameter'],str(file)))
        if(params['parameter']==parameter or params['parameter']=="None"):
            result.append(file)
    return result

def get_varied_params_from_file(param_file):
    myconfigparser = configparser.ConfigParser()
    myconfigparser.read(str(param_file))
    params = myconfigparser['params']
    try:
        varied_params_string = params['varied_params']
        varied_params = varied_params_string.split(' ')
    except:
        varied_params = all_varied_parameters
    return varied_params

def compute_param_files_list(varied_parameters,param_files):
    return [(param_files_with_parameter_variation(param_files,parameter=x),x) for x in varied_parameters]

def is_log_param(param):
    if param in use_log_parameter:
        return "--log_parameter"
    else:
        return ""

def deriv_cmd_string(param_files, obs_files, param):
    return ["-f %s %s %s" % (obs_file,parameter_value_this_param(param,x),is_log_param(param)) for x,obs_file in zip(param_files,obs_files)]

param_files_list = compute_param_files_list(all_varied_parameters, all_param_files)
param_files_list_centralsonly = compute_param_files_list(all_varied_parameters_centralsonly, all_param_files_centralsonly)

#### compute observables

## HOD param files
param_files = param_files_in_dir(param_dir)
param_files_centralsonly = param_files_in_dir_centralsonly(param_dir)

def observable_path(path, basepath="./Observables_0.1hMpc"):
    """change Path 'path' to be basepath / path.name"""
    return str(Path(basepath) / Path(path).name)

## compute observables from reconstructed xi_gg, xi_gm ##

def task_compute_wp():
    """compute Omega_M * xsi_gm (for now)"""

    for param_file in param_files:
        correlation_file = txt_reconstructed_xi_gg_this_param(param_file)
        subdir = subdir_from_param_file(param_file)
        header_file = header_file_this_sim(subdir)
        script = "./Analysis/compute_wp.py"
        deps = [correlation_file, header_file, script]
        targets = [observable_path(txt_reconstructed_wp_this_param(param_file))]
            
        yield {
            'actions': ["python %(script)s %(correlation_file)s %(header_file)s %(output)s --rpmin %(rmin)s --rpmax %(rmax)s --nbins %(nbins)s"
                        % {"script": script,
                           "correlation_file": correlation_file,
                           "header_file": header_file,
                           "output": targets[0],
                           "rmin": projected_rmin,
                           "rmax": projected_rmax,
                           "nbins": projected_nbins}],
            'file_dep': deps,
            'targets': targets,
            'name': str(correlation_file),
        }

def task_compute_delta_sigma():
    """compute DeltaSigma(r_p) from reconstructed xi_gm"""

    for param_file in param_files:
        correlation_file = txt_reconstructed_xi_gm_this_param(param_file)
        subdir = subdir_from_param_file(param_file)
        header_file = header_file_this_sim(subdir)
        script = "./Analysis/compute_DeltaSigma.py"
        deps = [correlation_file, header_file, script]
        targets = [observable_path(txt_reconstructed_DeltaSigma_this_param(param_file))]
            
        yield {
            'actions': ["python %(script)s %(correlation_file)s %(header_file)s %(output)s --rpmin %(rmin)s --rpmax %(rmax)s --nbins %(nbins)s"
                        % {"script": script,
                           "correlation_file": correlation_file,
                           "header_file": header_file,
                           "output": targets[0],
                           "rmin": projected_rmin,
                           "rmax": projected_rmax,
                           "nbins": projected_nbins}],
            'file_dep': deps,
            'targets': targets,
            'name': str(correlation_file),
        }

def task_compute_wp_centralsonly():
    """compute Omega_M * xsi_gm (for now)"""

    for param_file in param_files_centralsonly:
        correlation_file = txt_reconstructed_xi_gg_this_param_centralsonly(param_file)
        subdir = subdir_from_param_file(param_file)
        header_file = header_file_this_sim(subdir)
        script = "./Analysis/compute_wp.py"
        deps = [correlation_file, header_file, script]
        targets = [observable_path(txt_reconstructed_wp_this_param_centralsonly(param_file))]
            
        yield {
            'actions': ["python %(script)s %(correlation_file)s %(header_file)s %(output)s --rpmin %(rmin)s --rpmax %(rmax)s --nbins %(nbins)s"
                        % {"script": script,
                           "correlation_file": correlation_file,
                           "header_file": header_file,
                           "output": targets[0],
                           "rmin": projected_rmin,
                           "rmax": projected_rmax,
                           "nbins": projected_nbins}],
            'file_dep': deps,
            'targets': targets,
            'name': str(correlation_file),
        }

def task_compute_delta_sigma_centralsonly():
    """compute DeltaSigma(r_p) from reconstructed xi_gm"""

    for param_file in param_files_centralsonly:
        correlation_file = txt_reconstructed_xi_gm_this_param_centralsonly(param_file)
        subdir = subdir_from_param_file(param_file)
        header_file = header_file_this_sim(subdir)
        script = "./Analysis/compute_DeltaSigma.py"
        deps = [correlation_file, header_file, script]
        targets = [observable_path(txt_reconstructed_DeltaSigma_this_param_centralsonly(param_file))]
            
        yield {
            'actions': ["python %(script)s %(correlation_file)s %(header_file)s %(output)s --rpmin %(rmin)s --rpmax %(rmax)s --nbins %(nbins)s"
                        % {"script": script,
                           "correlation_file": correlation_file,
                           "header_file": header_file,
                           "output": targets[0],
                           "rmin": projected_rmin,
                           "rmax": projected_rmax,
                           "nbins": projected_nbins}],
            'file_dep': deps,
            'targets': targets,
            'name': str(correlation_file),
        }

### compute number density ###

def task_compute_galaxy_number_density():
    """compute number density of galaxies."""

    for param_file in param_files:
        mock_file = hdf5_HOD_mock_this_param(param_file)
        subdir = subdir_from_param_file(param_file)
        header = header_file_this_sim(subdir)
        script = "./Analysis/compute_number_density.py"
        deps = [mock_file, script, header]
        targets = [observable_path(txt_galaxy_number_density_this_param(param_file))]

        yield {
            'actions': ["python %(script)s %(header_file)s %(mock_file)s %(target)s"
                        % {"script": script,
                           "header_file": header, "mock_file": mock_file, "target": targets[0]}],
            'file_dep': deps,
            'targets': targets,
            'name': str(mock_file),
        }

def task_compute_galaxy_number_density_centralsonly():
    """compute number density of galaxies."""

    for param_file in param_files_centralsonly:
        mock_file = hdf5_HOD_mock_this_param_centralsonly(param_file)
        subdir = subdir_from_param_file(param_file)
        header = header_file_this_sim(subdir)
        script = "./Analysis/compute_number_density.py"
        deps = [mock_file, script, header]
        targets = [observable_path(txt_galaxy_number_density_this_param_centralsonly(param_file))]

        yield {
            'actions': ["python %(script)s --centrals_only %(header_file)s %(mock_file)s %(target)s"
                        % {"script": script,
                           "header_file": header, "mock_file": mock_file, "target": targets[0]}],
            'file_dep': deps,
            'targets': targets,
            'name': str(mock_file),
        }

## compute 'compensated' observables

def task_compute_DeltaSigma_baldauf():
    """compute DeltaSigma with small-scale info removed"""

    for param_file in param_files:
        correlation_file = txt_reconstructed_DeltaSigma_this_param(param_file)
        script = "./Analysis/compute_baldauf_estimator.py"
        deps = [correlation_file, script]
        targets = [observable_path(txt_reconstructed_DeltaSigma_baldauf_this_param(param_file))]
            
        yield {
            'actions': ["python %(script)s %(correlation_file)s %(output)s"
                        % {"script": script,
                           "correlation_file": correlation_file,
                           "output": targets[0]}],
            'file_dep': deps,
            'task_dep': ['compute_delta_sigma'],
            'targets': targets,
            'name': str(correlation_file),
        }

def task_compute_DeltaSigma_baldauf_centralsonly():
    """compute DeltaSigma with small-scale info removed"""

    for param_file in param_files_centralsonly:
        correlation_file = txt_reconstructed_DeltaSigma_this_param_centralsonly(param_file)
        script = "./Analysis/compute_baldauf_estimator.py"
        deps = [correlation_file, script]
        targets = [observable_path(txt_reconstructed_DeltaSigma_baldauf_this_param_centralsonly(param_file))]
            
        yield {
            'actions': ["python %(script)s %(correlation_file)s %(output)s"
                        % {"script": script,
                           "correlation_file": correlation_file,
                           "output": targets[0]}],
            'file_dep': deps,
            'task_dep': ['compute_delta_sigma_centralsonly'],
            'targets': targets,
            'name': str(correlation_file),
        }

## compute log observables

def task_compute_log_wp():
    """compute log w_p"""

    for param_file in param_files:
        correlation_file = txt_reconstructed_wp_this_param(param_file)
        script = "./Analysis/compute_ln_function.py"
        deps = [correlation_file, script]
        targets = [observable_path(txt_reconstructed_log_wp_this_param(param_file))]
            
        yield {
            'actions': ["python %(script)s %(correlation_file)s %(output)s"
                        % {"script": script,
                           "correlation_file": correlation_file,
                           "output": targets[0]}],
            'file_dep': deps,
            'task_dep': ['compute_wp'],
            'targets': targets,
            'name': str(correlation_file),
        }

def task_compute_log_DeltaSigma():
    """compute log DeltaSigma"""

    for param_file in param_files:
        correlation_file= txt_reconstructed_DeltaSigma_this_param(param_file)
        script = "./Analysis/compute_ln_function.py"
        deps = [correlation_file, script]
        targets = [observable_path(txt_reconstructed_log_DeltaSigma_this_param(param_file))]
            
        yield {
            'actions': ["python %(script)s %(correlation_file)s %(output)s"
                        % {"script": script,
                           "correlation_file": correlation_file,
                           "output": targets[0]}],
            'file_dep': deps,
            'task_dep': ['compute_delta_sigma'],
            'targets': targets,
            'name': str(correlation_file),
        }


def task_compute_log_wp_centralsonly():
    """compute log w_p"""

    for param_file in param_files_centralsonly:
        correlation_file = txt_reconstructed_wp_this_param_centralsonly(param_file)
        script = "./Analysis/compute_ln_function.py"
        deps = [correlation_file, script]
        targets = [observable_path(txt_reconstructed_log_wp_this_param_centralsonly(param_file))]
            
        yield {
            'actions': ["python %(script)s %(correlation_file)s %(output)s"
                        % {"script": script,
                           "correlation_file": correlation_file,
                           "output": targets[0]}],
            'file_dep': deps,
            'task_dep': ['compute_wp_centralsonly'],
            'targets': targets,
            'name': str(correlation_file),
        }

def task_compute_log_DeltaSigma_centralsonly():
    """compute log DeltaSigma"""

    for param_file in param_files_centralsonly:
        correlation_file = txt_reconstructed_DeltaSigma_this_param_centralsonly(param_file)
        script = "./Analysis/compute_ln_function.py"
        deps = [correlation_file, script]
        targets = [observable_path(txt_reconstructed_log_DeltaSigma_this_param_centralsonly(param_file))]
            
        yield {
            'actions': ["python %(script)s %(correlation_file)s %(output)s"
                        % {"script": script,
                           "correlation_file": correlation_file,
                           "output": targets[0]}],
            'file_dep': deps,
            'task_dep': ['compute_delta_sigma_centralsonly'],
            'targets': targets,
            'name': str(correlation_file),
        }


## compute log compensated observables

def task_compute_log_DeltaSigma_baldauf():
    """compute log DeltaSigma (Baldauf estimator) """

    for param_file in param_files:
        correlation_file = txt_reconstructed_DeltaSigma_baldauf_this_param(param_file)
        script = "./Analysis/compute_ln_function.py"
        deps = [correlation_file, script]
        targets = [observable_path(txt_reconstructed_log_DeltaSigma_baldauf_this_param(param_file))]
            
        yield {
            'actions': ["python %(script)s %(correlation_file)s %(output)s"
                        % {"script": script,
                           "correlation_file": correlation_file,
                           "output": targets[0]}],
            'file_dep': deps,
            'task_dep': ['compute_DeltaSigma_baldauf'],
            'targets': targets,
            'name': str(correlation_file),
        }

def task_compute_log_DeltaSigma_baldauf_centralsonly():
    """compute log DeltaSigma (Baldauf estimator) """

    for param_file in param_files_centralsonly:
        correlation_file = txt_reconstructed_DeltaSigma_baldauf_this_param_centralsonly(param_file)
        script = "./Analysis/compute_ln_function.py"
        deps = [correlation_file, script]
        targets = [observable_path(txt_reconstructed_log_DeltaSigma_baldauf_this_param_centralsonly(param_file))]
            
        yield {
            'actions': ["python %(script)s %(correlation_file)s %(output)s"
                        % {"script": script,
                           "correlation_file": correlation_file,
                           "output": targets[0]}],
            'file_dep': deps,
            'task_dep': ['compute_DeltaSigma_baldauf_centralsonly'],
            'targets': targets,
            'name': str(correlation_file),
        }

## compute log number density

def task_compute_log_number_density():
    """compute log number density"""

    for param_file in param_files:
        correlation_file = txt_galaxy_number_density_this_param(param_file)
        script = "./Analysis/compute_ln_array.py"
        deps = [correlation_file, script]
        targets = [observable_path(txt_log_galaxy_number_density_this_param(param_file))]
            
        yield {
            'actions': ["python %(script)s %(correlation_file)s %(output)s"
                        % {"script": script,
                           "correlation_file": correlation_file,
                           "output": targets[0]}],
            'file_dep': deps,
            'task_dep': ['compute_galaxy_number_density'],
            'targets': targets,
            'name': str(correlation_file),
        }

def task_compute_log_number_density_centralsonly():
    """compute log number density"""

    for param_file in param_files_centralsonly:
        correlation_file = txt_galaxy_number_density_this_param_centralsonly(param_file)
        script = "./Analysis/compute_ln_array.py"
        deps = [correlation_file, script]
        targets = [observable_path(txt_log_galaxy_number_density_this_param_centralsonly(param_file))]
            
        yield {
            'actions': ["python %(script)s %(correlation_file)s %(output)s"
                        % {"script": script,
                           "correlation_file": correlation_file,
                           "output": targets[0]}],
            'file_dep': deps,
            'task_dep': ['compute_galaxy_number_density_centralsonly'],
            'targets': targets,
            'name': str(correlation_file),
        }


## compute derivatives of log observables

def task_compute_derivative_log_wp():
    """compute derivative of w_p"""
    for param_files, param in param_files_list:
        correlation_files = [observable_path(txt_reconstructed_log_wp_this_param(x)) for x in param_files]
        input_string = deriv_cmd_string(param_files, correlation_files, param)
        script = "./Analysis/compute_partial_derivative.py"
        deps = correlation_files+[script]
        output = './Derivatives_0.1hMpc/ln_wp_%s.txt' % param
        targets = [output]
        
        yield {
            'actions': ["python %(script)s %(output)s %(inputs)s"
                        % {"script": script, "output": output, "inputs": ' '.join(input_string)}],
            'file_dep': deps,
            'targets': targets,
            'name': working_directory / param
        }

def task_compute_derivative_log_DeltaSigma():
    """compute derivative of \Delta\Sigma"""
    for param_files, param in param_files_list:
        correlation_files = [observable_path(txt_reconstructed_log_DeltaSigma_this_param(x)) for x in param_files]
        input_string = deriv_cmd_string(param_files, correlation_files, param)
        script = "./Analysis/compute_partial_derivative.py"
        deps = correlation_files+[script]
        output = './Derivatives_0.1hMpc/ln_DeltaSigma_%s.txt' % param
        targets = [output]
        
        yield {
            'actions': ["python %(script)s %(output)s %(inputs)s"
                        % {"script": script, "output": output, "inputs": ' '.join(input_string)}],
            'file_dep': deps,
            'targets': targets,
            'name': working_directory / param
        }

def task_compute_derivative_log_DeltaSigma_baldauf():
    """compute derivative of \Delta\Sigma"""
    for param_files, param in param_files_list:
        correlation_files = [observable_path(txt_reconstructed_log_DeltaSigma_baldauf_this_param(x)) for x in param_files]
        input_string = deriv_cmd_string(param_files, correlation_files, param)
        script = "./Analysis/compute_partial_derivative.py"
        deps = correlation_files+[script]
        output = './Derivatives_0.1hMpc/ln_DeltaSigma_baldauf_%s.txt' % param
        targets = [output]
        
        yield {
            'actions': ["python %(script)s %(output)s %(inputs)s"
                        % {"script": script, "output": output, "inputs": ' '.join(input_string)}],
            'file_dep': deps,
            'targets': targets,
            'name': working_directory / param
        }

def task_compute_derivative_log_DeltaSigma_baldauf_centralsonly():
    """compute derivative of \Delta\Sigma"""
    for param_files, param in param_files_list_centralsonly:
        correlation_files = [observable_path(txt_reconstructed_log_DeltaSigma_baldauf_this_param_centralsonly(x)) for x in param_files]
        input_string = deriv_cmd_string(param_files, correlation_files, param)
        script = "./Analysis/compute_partial_derivative.py"
        deps = correlation_files+[script]
        output = './Derivatives_0.1hMpc/ln_DeltaSigma_baldauf_%s.centralsonly.txt' % param
        targets = [output]
        
        yield {
            'actions': ["python %(script)s %(output)s %(inputs)s"
                        % {"script": script, "output": output, "inputs": ' '.join(input_string)}],
            'file_dep': deps,
            'targets': targets,
            'name': working_directory / param
        }

def task_compute_derivative_log_ngal():
    """compute derivative of log ngal"""
    for param_files, param in param_files_list:
        correlation_files = [observable_path(txt_log_galaxy_number_density_this_param(x)) for x in param_files]
        input_string = deriv_cmd_string(param_files, correlation_files, param)
        script = "./Analysis/compute_partial_derivative_array.py"
        deps = correlation_files+[script]
        output = './Derivatives_0.1hMpc/ln_ngal_%s.txt' % param
        targets = [output]
        
        yield {
            'actions': ["python %(script)s %(output)s %(inputs)s"
                        % {"script": script, "output": output, "inputs": ' '.join(input_string)}],
            'file_dep': deps,
            'targets': targets,
            'name': working_directory / param
        }

def task_compute_derivative_log_wp_centralsonly():
    """compute derivative of w_p"""
    for param_files, param in param_files_list_centralsonly:
        correlation_files = [observable_path(txt_reconstructed_log_wp_this_param_centralsonly(x)) for x in param_files]
        input_string = deriv_cmd_string(param_files, correlation_files, param)
        script = "./Analysis/compute_partial_derivative.py"
        deps = correlation_files+[script]
        output = './Derivatives_0.1hMpc/ln_wp_%s.centralsonly.txt' % param
        targets = [output]
        
        yield {
            'actions': ["python %(script)s %(output)s %(inputs)s"
                        % {"script": script, "output": output, "inputs": ' '.join(input_string)}],
            'file_dep': deps,
            'targets': targets,
            'name': working_directory / param
        }

def task_compute_derivative_log_DeltaSigma_centralsonly():
    """compute derivative of \Delta\Sigma"""
    for param_files, param in param_files_list_centralsonly:
        correlation_files = [observable_path(txt_reconstructed_log_DeltaSigma_this_param_centralsonly(x)) for x in param_files]
        input_string = deriv_cmd_string(param_files, correlation_files, param)
        script = "./Analysis/compute_partial_derivative.py"
        deps = correlation_files+[script]
        output = './Derivatives_0.1hMpc/ln_DeltaSigma_%s.centralsonly.txt' % param
        targets = [output]
        
        yield {
            'actions': ["python %(script)s %(output)s %(inputs)s"
                        % {"script": script, "output": output, "inputs": ' '.join(input_string)}],
            'file_dep': deps,
            'targets': targets,
            'name': working_directory / param
        }

def task_compute_derivative_log_ngal_centralsonly():
    """compute derivative of log ngal"""
    for param_files, param in param_files_list_centralsonly:
        correlation_files = [observable_path(txt_log_galaxy_number_density_this_param_centralsonly(x)) for x in param_files]
        input_string = deriv_cmd_string(param_files, correlation_files, param)
        script = "./Analysis/compute_partial_derivative_array.py"
        deps = correlation_files+[script]
        output = './Derivatives_0.1hMpc/ln_ngal_%s.centralsonly.txt' % param
        targets = [output]
        
        yield {
            'actions': ["python %(script)s --debug %(output)s %(inputs)s"
                        % {"script": script, "output": output, "inputs": ' '.join(input_string)}],
            'file_dep': deps,
            'targets': targets,
            'name': working_directory / param
        }


## compute derivatives of observables

def task_compute_derivative_wp():
    """compute derivative of w_p"""
    for param_files, param in param_files_list:
        correlation_files = [observable_path(txt_reconstructed_wp_this_param(x)) for x in param_files]
        input_string = deriv_cmd_string(param_files, correlation_files, param)
        script = "./Analysis/compute_partial_derivative.py"
        deps = correlation_files+[script]
        output = './Derivatives_0.1hMpc/wp_%s.txt' % param
        targets = [output]
        
        yield {
            'actions': ["python %(script)s %(output)s %(inputs)s"
                        % {"script": script, "output": output, "inputs": ' '.join(input_string)}],
            'file_dep': deps,
            'targets': targets,
            'name': working_directory / param
        }

def task_compute_derivative_DeltaSigma():
    """compute derivative of \Delta\Sigma"""
    for param_files, param in param_files_list:
        correlation_files = [observable_path(txt_reconstructed_DeltaSigma_this_param(x)) for x in param_files]
        input_string = deriv_cmd_string(param_files, correlation_files, param)
        script = "./Analysis/compute_partial_derivative.py"
        deps = correlation_files+[script]
        output = './Derivatives_0.1hMpc/DeltaSigma_%s.txt' % param
        targets = [output]
        
        yield {
            'actions': ["python %(script)s %(output)s %(inputs)s"
                        % {"script": script, "output": output, "inputs": ' '.join(input_string)}],
            'file_dep': deps,
            'targets': targets,
            'name': working_directory / param
        }

def task_compute_derivative_wp_centralsonly():
    """compute derivative of w_p"""
    for param_files, param in param_files_list_centralsonly:
        correlation_files = [observable_path(txt_reconstructed_wp_this_param_centralsonly(x)) for x in param_files]
        input_string = deriv_cmd_string(param_files, correlation_files, param)
        script = "./Analysis/compute_partial_derivative.py"
        deps = correlation_files+[script]
        output = './Derivatives_0.1hMpc/wp_%s.centralsonly.txt' % param
        targets = [output]
        
        yield {
            'actions': ["python %(script)s %(output)s %(inputs)s"
                        % {"script": script, "output": output, "inputs": ' '.join(input_string)}],
            'file_dep': deps,
            'targets': targets,
            'name': working_directory / param
        }

def task_compute_derivative_DeltaSigma_centralsonly():
    """compute derivative of \Delta\Sigma"""
    for param_files, param in param_files_list_centralsonly:
        correlation_files = [observable_path(txt_reconstructed_DeltaSigma_this_param_centralsonly(x)) for x in param_files]
        input_string = deriv_cmd_string(param_files, correlation_files, param)
        script = "./Analysis/compute_partial_derivative.py"
        deps = correlation_files+[script]
        output = './Derivatives_0.1hMpc/DeltaSigma_%s.centralsonly.txt' % param
        targets = [output]
        
        yield {
            'actions': ["python %(script)s %(output)s %(inputs)s"
                        % {"script": script, "output": output, "inputs": ' '.join(input_string)}],
            'file_dep': deps,
            'targets': targets,
            'name': working_directory / param
        }

## compute pk of fiducial (for covariance matrix)

def task_compute_pk_mm():
    """compute matter power spectrum of fiducial case"""
    subdir = halo_working_directory / 'fiducial' / redshift
    power_subdir = str(subdir).replace("Rockstar", "power") #particle subsamples are in the corresponding FOF directory
    power_spectrum = str(power_subdir / Path("../info/camb_matterpower.dat"))
    script = "./Analysis/compute_pk.py"
    correlation_file = txt_matter_autocorrelation_this_sim(Path(subdir))
    deps = [correlation_file, script]
    targets = ["./Covariances_0.1hMpc/pk_mm.txt"]
            
    yield {
        'actions': ["python %(script)s %(correlation_file)s %(output)s %(linear_pk)s"
                    % {"script": script,
                       "correlation_file": correlation_file,
                       "linear_pk": power_spectrum,
                       "output": targets[0]}],
        'file_dep': deps,
        'targets': targets,
        'name': str(correlation_file),
    }

def task_compute_pk_gg():
    """compute galaxy power spectrum of fiducial case"""
    subdir = halo_working_directory / 'fiducial' / redshift
    power_subdir = str(subdir).replace("Rockstar", "power") #particle subsamples are in the corresponding FOF directory
    power_spectrum = str(power_subdir / Path("../info/camb_matterpower.dat"))
    script = "./Analysis/compute_pk.py"
    correlation_file = txt_reconstructed_xi_gg_this_param(Path(param_dir) / 'NHOD_fiducial.template_param')
    deps = [correlation_file, script]
    targets = ["./Covariances_0.1hMpc/pk_gg.txt"]
            
    yield {
        'actions': ["python %(script)s %(correlation_file)s %(output)s %(linear_pk)s"
                    % {"script": script,
                       "correlation_file": correlation_file,
                       "linear_pk": power_spectrum,
                       "output": targets[0]}],
        'file_dep': deps,
        'targets': targets,
        'name': str(correlation_file),
    }

def task_compute_pk_gm():
    """compute galaxy power spectrum of fiducial case"""
    subdir = halo_working_directory / 'fiducial' / redshift
    power_subdir = str(subdir).replace("Rockstar", "power") #particle subsamples are in the corresponding FOF directory
    power_spectrum = str(power_subdir / Path("../info/camb_matterpower.dat"))
    script = "./Analysis/compute_pk.py"
    correlation_file = txt_reconstructed_xi_gm_this_param(Path(param_dir) / 'NHOD_fiducial.template_param')
    deps = [correlation_file, script]
    targets = ["./Covariances_0.1hMpc/pk_gm.txt"]
            
    yield {
        'actions': ["python %(script)s %(correlation_file)s %(output)s %(linear_pk)s"
                    % {"script": script,
                       "correlation_file": correlation_file,
                       "linear_pk": power_spectrum,
                       "output": targets[0]}],
        'file_dep': deps,
        'targets': targets,
        'name': str(correlation_file),
    }

def task_compute_pk_gg_centralsonly():
    """compute galaxy power spectrum of fiducial case"""
    subdir = halo_working_directory / 'fiducial' / redshift
    power_subdir = str(subdir).replace("Rockstar", "power") #particle subsamples are in the corresponding FOF directory
    power_spectrum = str(power_subdir / Path("../info/camb_matterpower.dat"))
    script = "./Analysis/compute_pk.py"
    correlation_file = txt_reconstructed_xi_gg_this_param_centralsonly(Path(param_dir) / 'NHOD_fiducial.template_param_centralsonly')
    deps = [correlation_file, script]
    targets = ["./Covariances_0.1hMpc/pk_gg.centralsonly.txt"]
            
    yield {
        'actions': ["python %(script)s %(correlation_file)s %(output)s %(linear_pk)s"
                    % {"script": script,
                       "correlation_file": correlation_file,
                       "linear_pk": power_spectrum,
                       "output": targets[0]}],
        'file_dep': deps,
        'targets': targets,
        'name': str(correlation_file),
    }

def task_compute_pk_gm_centralsonly():
    """compute galaxy power spectrum of fiducial case"""
    subdir = halo_working_directory / 'fiducial' / redshift
    power_subdir = str(subdir).replace("Rockstar", "power") #particle subsamples are in the corresponding FOF directory
    power_spectrum = str(power_subdir / Path("../info/camb_matterpower.dat"))
    script = "./Analysis/compute_pk.py"
    correlation_file = txt_reconstructed_xi_gm_this_param_centralsonly(Path(param_dir) / 'NHOD_fiducial.template_param_centralsonly')
    deps = [correlation_file, script]
    targets = ["./Covariances_0.1hMpc/pk_gm.centralsonly.txt"]
            
    yield {
        'actions': ["python %(script)s %(correlation_file)s %(output)s %(linear_pk)s"
                    % {"script": script,
                       "correlation_file": correlation_file,
                       "linear_pk": power_spectrum,
                       "output": targets[0]}],
        'file_dep': deps,
        'targets': targets,
        'name': str(correlation_file),
    }

## compute covariance matrix

def task_compute_covariance_wp():
    script = './Analysis/compute_covar_wp.py'
    pk_gg = './Covariances_0.1hMpc/pk_gg.txt'
    param_files = Path('./Covariances_0.1hMpc').glob('*.param')

    for param_file in param_files:
        deps = [script, pk_gg, str(param_file)]
        covar = str(param_file.with_suffix('.wp.covar'))
        precision = str(param_file.with_suffix('.wp.precision'))
        signal = str(param_file.with_suffix('.wp.signal'))
        targets = [covar, precision, signal]

        yield {
            'actions': ["python %(script)s %(param)s %(pk_gg)s %(output_covar)s %(output_precision)s %(output_signal)s"
                        % {"script": script,
                           "param": param_file,
                           "pk_gg": pk_gg,
                           "output_covar": covar,
                           "output_precision": precision,
                           "output_signal": signal,
                       }],
            'file_dep': deps,
            'targets': targets,
            'name': param_file,
        }

def task_compute_covariance_DeltaSigma():
    script = './Analysis/compute_covar_DeltaSigma.py'
    pk_gg = './Covariances_0.1hMpc/pk_gg.txt'
    pk_gm = './Covariances_0.1hMpc/pk_gm.txt'
    pk_mm = './Covariances_0.1hMpc/pk_mm.txt'
    param_files = Path('./Covariances_0.1hMpc').glob('*.param')

    for param_file in param_files:
        deps = [script, pk_gg, pk_gm, pk_mm, str(param_file)]
        covar = str(param_file.with_suffix('.DeltaSigma.covar'))
        precision = str(param_file.with_suffix('.DeltaSigma.precision'))
        signal = str(param_file.with_suffix('.DeltaSigma.signal'))
        targets = [covar, precision, signal]

        yield {
            'actions': ["python %(script)s %(param)s %(pk_gg)s %(pk_gm)s %(pk_mm)s %(output_covar)s %(output_precision)s %(output_signal)s"
                        % {"script": script,
                           "param": param_file,
                           "pk_gg": pk_gg,
                           "pk_gm": pk_gm,
                           "pk_mm": pk_mm,
                           "output_covar": covar,
                           "output_precision": precision,
                           "output_signal": signal,
                       }],
            'file_dep': deps,
            'targets': targets,
            'name': param_file,
        }

def task_compute_covariance_wp_centralsonly():
    script = './Analysis/compute_covar_wp.py'
    pk_gg = './Covariances_0.1hMpc/pk_gg.centralsonly.txt'
    param_files = Path('./Covariances_0.1hMpc').glob('*.param')

    for param_file in param_files:
        deps = [script, pk_gg, str(param_file)]
        covar = str(param_file.with_suffix('.centralsonly.wp.covar'))
        precision = str(param_file.with_suffix('.centralsonly.wp.precision'))
        signal = str(param_file.with_suffix('.centralsonly.wp.signal'))
        targets = [covar, precision, signal]

        yield {
            'actions': ["python %(script)s %(param)s %(pk_gg)s %(output_covar)s %(output_precision)s %(output_signal)s"
                        % {"script": script,
                           "param": param_file,
                           "pk_gg": pk_gg,
                           "output_covar": covar,
                           "output_precision": precision,
                           "output_signal": signal,
                       }],
            'file_dep': deps,
            'targets': targets,
            'name': param_file,
        }

def task_compute_covariance_DeltaSigma_centralsonly():
    script = './Analysis/compute_covar_DeltaSigma.py'
    pk_gg = './Covariances_0.1hMpc/pk_gg.centralsonly.txt'
    pk_gm = './Covariances_0.1hMpc/pk_gm.centralsonly.txt'
    pk_mm = './Covariances_0.1hMpc/pk_mm.txt'
    param_files = Path('./Covariances_0.1hMpc').glob('*.param')

    for param_file in param_files:
        deps = [script, pk_gg, pk_gm, pk_mm, str(param_file)]
        covar = str(param_file.with_suffix('.centralsonly.DeltaSigma.covar'))
        precision = str(param_file.with_suffix('.centralsonly.DeltaSigma.precision'))
        signal = str(param_file.with_suffix('.centralsonly.DeltaSigma.signal'))
        targets = [covar, precision, signal]

        yield {
            'actions': ["python %(script)s %(param)s %(pk_gg)s %(pk_gm)s %(pk_mm)s %(output_covar)s %(output_precision)s %(output_signal)s"
                        % {"script": script,
                           "param": param_file,
                           "pk_gg": pk_gg,
                           "pk_gm": pk_gm,
                           "pk_mm": pk_mm,
                           "output_covar": covar,
                           "output_precision": precision,
                           "output_signal": signal,
                       }],
            'file_dep': deps,
            'targets': targets,
            'name': param_file,
        }

def task_compute_fractional_covariance_wp():
    script = './Analysis/compute_fractional_covar.py'
    param_files = Path('./Covariances_0.1hMpc').glob('*.param_frac')

    for param_file in param_files:
        deps = [script, str(param_file)]
        covar = str(param_file.with_suffix('.wp.covar_frac'))
        targets = [covar]

        yield {
            'actions': ["python %(script)s --observable wp %(param)s %(output_covar)s"
                        % {"script": script,
                           "param": param_file,
                           "output_covar": covar,
                       }],
            'file_dep': deps,
            'targets': targets,
            'name': param_file,
        }

def task_compute_fractional_covariance_DeltaSigma():
    script = './Analysis/compute_fractional_covar.py'
    param_files = Path('./Covariances_0.1hMpc').glob('*.param_frac')

    for param_file in param_files:
        deps = [script, str(param_file)]
        covar = str(param_file.with_suffix('.DeltaSigma.covar_frac'))
        targets = [covar]

        yield {
            'actions': ["python %(script)s --observable DS %(param)s %(output_covar)s"
                        % {"script": script,
                           "param": param_file,
                           "output_covar": covar,
                       }],
            'file_dep': deps,
            'targets': targets,
            'name': param_file,
        }

def task_compute_covariance_ngal():
    script = './Analysis/compute_fractional_covar.py'
    param_files = Path('./Covariances_0.1hMpc').glob('*.param')

    for param_file in param_files:
        deps = [script, str(param_file)]
        covar = str(param_file.with_suffix('.ngal.covar'))
        targets = [covar]

        yield {
            'actions': ["python %(script)s --observable ngal %(param)s %(output_covar)s"
                        % {"script": script,
                           "param": param_file,
                           "output_covar": covar,
                       }],
            'file_dep': deps,
            'targets': targets,
            'name': param_file,
        }

def task_compute_covariance_ngal_centralsonly():
    script = './Analysis/compute_fractional_covar.py'
    param_files = Path('./Covariances_0.1hMpc').glob('*.param')

    for param_file in param_files:
        deps = [script, str(param_file)]
        covar = str(param_file.with_suffix('.centralsonly.ngal.covar'))
        targets = [covar]

        yield {
            'actions': ["python %(script)s --observable ngal %(param)s %(output_covar)s"
                        % {"script": script,
                           "param": param_file,
                           "output_covar": covar,
                       }],
            'file_dep': deps,
            'targets': targets,
            'name': param_file,
        }

def task_compute_fractional_covariance_ngal():
    script = './Analysis/compute_fractional_covar.py'
    param_files = Path('./Covariances_0.1hMpc').glob('*.param_frac')

    for param_file in param_files:
        deps = [script, str(param_file)]
        covar = str(param_file.with_suffix('.ngal.covar_frac'))
        targets = [covar]

        yield {
            'actions': ["python %(script)s --observable ngal %(param)s %(output_covar)s"
                        % {"script": script,
                           "param": param_file,
                           "output_covar": covar,
                       }],
            'file_dep': deps,
            'targets': targets,
            'name': param_file,
        }


## plot covariance matrices

def task_plot_covariance_wp():
    script = './Plotting/plot_covar.py'
    param_files = Path('./Covariances_0.1hMpc').glob('*.param')

    for param_file in param_files:
        covar = str(param_file.with_suffix('.wp.covar'))
        signal = str(param_file.with_suffix('.wp.signal'))
        covar_plot = str(param_file.with_suffix('.wp.covar.pdf'))
        signal_plot = str(param_file.with_suffix('.wp.signal.pdf'))
        deps = [script, covar, signal]
        targets = [covar_plot, signal_plot]

        yield {
            'actions': ["python %(script)s %(covar)s %(signal)s %(covar_plot)s %(signal_plot)s '%(title)s'"
                        % {"script": script,
                           "covar": covar,
                           "signal": signal,
                           "covar_plot": covar_plot,
                           "signal_plot": signal_plot,
                           "title": "$w_p$",
                       }],
            'file_dep': deps,
            'targets': targets,
            'name': param_file,
        }

def task_plot_covariance_DeltaSigma():
    script = './Plotting/plot_covar.py'
    param_files = Path('./Covariances_0.1hMpc').glob('*.param')

    for param_file in param_files:
        covar = str(param_file.with_suffix('.DeltaSigma.covar'))
        signal = str(param_file.with_suffix('.DeltaSigma.signal'))
        covar_plot = str(param_file.with_suffix('.DeltaSigma.covar.pdf'))
        signal_plot = str(param_file.with_suffix('.DeltaSigma.signal.pdf'))
        deps = [script, covar, signal]
        targets = [covar_plot, signal_plot]

        yield {
            'actions': ["python %(script)s %(covar)s %(signal)s %(covar_plot)s %(signal_plot)s '%(title)s'"
                        % {"script": script,
                           "covar": covar,
                           "signal": signal,
                           "covar_plot": covar_plot,
                           "signal_plot": signal_plot,
                           "title": "$\gamma_t$",
                       }],
            'file_dep': deps,
            'targets': targets,
            'name': param_file,
        }



## compute Fisher matrices
## TODO: read parameters from param_files

def task_compute_fisher_matrix_DeltaSigma():
    """compute Fisher matrix for DeltaSigma"""
    script = "./Analysis/compute_fisher_matrix.py"
    param_files = Path('./Covariances_0.1hMpc').glob('*.param')

    for param_file in param_files:
        varied_params = get_varied_params_from_file(param_file)
        param_files_list = compute_param_files_list(varied_params,all_param_files)
        derivative_files = [('./Derivatives_0.1hMpc/DeltaSigma_%s.txt' % param, param) for param_files, param in param_files_list]
        obs_files, params = zip(*derivative_files)
        input_string = ["-f %s %s" % file_tuple for file_tuple in derivative_files]

        fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".DeltaSigma.fisher"
        input_cov_matrix_file = str(param_file.with_suffix('.DeltaSigma.covar'))
        bins_file = str(param_file.with_suffix('.DeltaSigma.signal'))
        deps = obs_files+(script,input_cov_matrix_file)
        targets = [fisher_matrix_file]
    
        cmd_string = "python %(script)s %(output)s %(input_cov)s --bins_file %(bins_file)s --parameter_file %(param_file)s --observable DS %(inputs)s" \
                     % {"script": script,
                        "output": fisher_matrix_file,
                        "input_cov": input_cov_matrix_file,
                        "bins_file": bins_file,
                        "param_file": param_file,
                        "inputs": ' '.join(input_string)}

        yield {
            'actions': [cmd_string],
            'file_dep': deps,
            'task_dep': ['compute_derivative_DeltaSigma'],
            'targets': targets,
            'name': param_file
        }

def task_compute_fisher_matrix_wp():
    """compute Fisher matrix for w_p"""
    script = "./Analysis/compute_fisher_matrix.py"
    param_files = Path('./Covariances_0.1hMpc').glob('*.param')

    for param_file in param_files:
        varied_params = get_varied_params_from_file(param_file)
        param_files_list = compute_param_files_list(varied_params,all_param_files)
        derivative_files = [('./Derivatives_0.1hMpc/wp_%s.txt' % param, param) for param_files, param in param_files_list]
        obs_files, params = zip(*derivative_files)
        input_string = ["-f %s %s" % file_tuple for file_tuple in derivative_files]

        fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".wp.fisher"
        input_cov_matrix_file = str(param_file.with_suffix('.wp.covar'))
        bins_file = str(param_file.with_suffix('.wp.signal'))
        deps = obs_files+(script,input_cov_matrix_file)
        targets = [fisher_matrix_file]
    
        cmd_string = "python %(script)s %(output)s %(input_cov)s --bins_file %(bins_file)s --parameter_file %(param_file)s --observable wp %(inputs)s" \
                     % {"script": script,
                        "output": fisher_matrix_file,
                        "input_cov": input_cov_matrix_file,
                        "bins_file": bins_file,
                        "param_file": param_file,
                        "inputs": ' '.join(input_string)}

        yield {
            'actions': [cmd_string],
            'file_dep': deps,
            'task_dep': ['compute_derivative_wp'],
            'targets': targets,
            'name': param_file
        }

def task_compute_fisher_matrix_ngal():
    """compute Fisher matrix for n_gal"""
    script = "./Analysis/compute_fisher_matrix.py"
    param_files = Path('./Covariances_0.1hMpc').glob('*.param')

    for param_file in param_files:
        varied_params = get_varied_params_from_file(param_file)
        param_files_list = compute_param_files_list(varied_params,all_param_files)
        derivative_files = [('./Derivatives_0.1hMpc/ln_ngal_%s.txt' % param, param) for param_files, param in param_files_list]
        input_string = ["-f %s %s" % file_tuple for file_tuple in derivative_files]

        obs_files, params = zip(*derivative_files)
        fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".ngal.fisher"
        input_cov_matrix_file = str(param_file.with_suffix('.ngal.covar'))
        deps = obs_files+(script,input_cov_matrix_file)
        targets = [fisher_matrix_file]
    
        cmd_string = "python %(script)s %(output)s %(input_cov)s %(inputs)s" \
                     % {"script": script,
                        "output": fisher_matrix_file,
                        "input_cov": input_cov_matrix_file,
                        "inputs": ' '.join(input_string)}

        yield {
            'actions': [cmd_string],
            'file_dep': deps,
            'targets': targets,
            'name': param_file
        }

def task_compute_fisher_matrix_DeltaSigma_centralsonly():
    """compute Fisher matrix for DeltaSigma"""
    script = "./Analysis/compute_fisher_matrix.py"
    param_files = Path('./Covariances_0.1hMpc').glob('*.param')

    for param_file in param_files:
        varied_params = get_varied_params_from_file(param_file)
#        param_files_list = compute_param_files_list(varied_params,all_param_files_centralsonly)
        param_files_list = param_files_list_centralsonly
        derivative_files = [('./Derivatives_0.1hMpc/DeltaSigma_%s.centralsonly.txt' % param, param) for param_files, param in param_files_list]
        obs_files, params = zip(*derivative_files)
        input_string = ["-f %s %s" % file_tuple for file_tuple in derivative_files]

        fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".centralsonly.DeltaSigma.fisher"
        input_cov_matrix_file = str(param_file.with_suffix('.centralsonly.DeltaSigma.covar'))
        bins_file = str(param_file.with_suffix('.centralsonly.DeltaSigma.signal'))
        deps = obs_files+(script,input_cov_matrix_file)
        targets = [fisher_matrix_file]
    
        cmd_string = "python %(script)s %(output)s %(input_cov)s --bins_file %(bins_file)s --parameter_file %(param_file)s --observable DS %(inputs)s" \
                     % {"script": script,
                        "output": fisher_matrix_file,
                        "input_cov": input_cov_matrix_file,
                        "bins_file": bins_file,
                        "param_file": param_file,
                        "inputs": ' '.join(input_string)}

        yield {
            'actions': [cmd_string],
            'file_dep': deps,
            'task_dep': ['compute_derivative_DeltaSigma_centralsonly'],
            'targets': targets,
            'name': param_file
        }

def task_compute_fisher_matrix_wp_centralsonly():
    """compute Fisher matrix for w_p"""
    script = "./Analysis/compute_fisher_matrix.py"
    param_files = Path('./Covariances_0.1hMpc').glob('*.param')

    for param_file in param_files:
        varied_params = get_varied_params_from_file(param_file)
#        param_files_list = compute_param_files_list(varied_params,all_param_files_centralsonly)
        param_files_list = param_files_list_centralsonly
        derivative_files = [('./Derivatives_0.1hMpc/wp_%s.centralsonly.txt' % param, param) for param_files, param in param_files_list]
        obs_files, params = zip(*derivative_files)
        input_string = ["-f %s %s" % file_tuple for file_tuple in derivative_files]

        fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".centralsonly.wp.fisher"
        input_cov_matrix_file = str(param_file.with_suffix('.centralsonly.wp.covar'))
        bins_file = str(param_file.with_suffix('.centralsonly.wp.signal'))
        deps = obs_files+(script,input_cov_matrix_file)
        targets = [fisher_matrix_file]
    
        cmd_string = "python %(script)s %(output)s %(input_cov)s --bins_file %(bins_file)s --parameter_file %(param_file)s --observable wp %(inputs)s" \
                     % {"script": script,
                        "output": fisher_matrix_file,
                        "input_cov": input_cov_matrix_file,
                        "bins_file": bins_file,
                        "param_file": param_file,
                        "inputs": ' '.join(input_string)}

        yield {
            'actions': [cmd_string],
            'file_dep': deps,
            'task_dep': ['compute_derivative_wp_centralsonly'],
            'targets': targets,
            'name': param_file
        }

def task_compute_fisher_matrix_ngal_centralsonly():
    """compute Fisher matrix for n_gal"""
    script = "./Analysis/compute_fisher_matrix.py"
    param_files = Path('./Covariances_0.1hMpc').glob('*.param')

    for param_file in param_files:
        varied_params = get_varied_params_from_file(param_file)
        param_files_list = param_files_list_centralsonly 
        derivative_files = [('./Derivatives_0.1hMpc/ln_ngal_%s.centralsonly.txt' % param, param) for param_files, param in param_files_list]
        input_string = ["-f %s %s" % file_tuple for file_tuple in derivative_files]

        obs_files, params = zip(*derivative_files)
        fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".centralsonly.ngal.fisher"
        input_cov_matrix_file = str(param_file.with_suffix('.centralsonly.ngal.covar'))
        deps = obs_files+(script,input_cov_matrix_file)
        targets = [fisher_matrix_file]
    
        cmd_string = "python %(script)s %(output)s %(input_cov)s %(inputs)s" \
                     % {"script": script,
                        "output": fisher_matrix_file,
                        "input_cov": input_cov_matrix_file,
                        "inputs": ' '.join(input_string)}

        yield {
            'actions': [cmd_string],
            'file_dep': deps,
            'targets': targets,
            'name': param_file
        }

# def task_compute_fisher_matrix_DeltaSigma_frac():
#     """compute Fisher matrix for DeltaSigma using fractional covariances"""
#     script = "./Analysis/compute_fisher_matrix.py"
#     param_files = Path('./Covariances_0.1hMpc').glob('*.param_frac')

#     for param_file in param_files:
#         varied_params = get_varied_params_from_file(param_file)
#         param_files_list = compute_param_files_list(varied_params,all_param_files)
#         derivative_files = [('./Derivatives_0.1hMpc/ln_DeltaSigma_%s.txt' % param, param) for param_files, param in param_files_list]
#         obs_files, params = zip(*derivative_files)
#         input_string = ["-f %s %s" % file_tuple for file_tuple in derivative_files]

#         fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".DeltaSigma.fisher_frac"
#         input_cov_matrix_file = str(param_file.with_suffix('.DeltaSigma.covar_frac'))
#         bins_file = str(param_file.with_suffix('.DeltaSigma.signal'))
#         deps = obs_files+(script,input_cov_matrix_file)
#         targets = [fisher_matrix_file]
    
#         cmd_string = "python %(script)s %(output)s %(input_cov)s --bins_file %(bins_file)s --parameter_file %(param_file)s --observable DS %(inputs)s" \
#                      % {"script": script,
#                         "output": fisher_matrix_file,
#                         "input_cov": input_cov_matrix_file,
#                         "bins_file": bins_file,
#                         "param_file": param_file,
#                         "inputs": ' '.join(input_string)}

#         yield {
#             'actions': [cmd_string],
#             'file_dep': deps,
#             'task_dep': ['compute_derivative_DeltaSigma'],
#             'targets': targets,
#             'name': param_file
#         }

# def task_compute_fisher_matrix_wp_frac():
#     """compute Fisher matrix for w_p using fractional covariances"""
#     script = "./Analysis/compute_fisher_matrix.py"
#     param_files = Path('./Covariances_0.1hMpc').glob('*.param_frac')

#     for param_file in param_files:
#         varied_params = get_varied_params_from_file(param_file)
#         param_files_list = compute_param_files_list(varied_params,all_param_files)
#         derivative_files = [('./Derivatives_0.1hMpc/ln_wp_%s.txt' % param, param) for param_files, param in param_files_list]
#         obs_files, params = zip(*derivative_files)
#         input_string = ["-f %s %s" % file_tuple for file_tuple in derivative_files]

#         fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".wp.fisher_frac"
#         input_cov_matrix_file = str(param_file.with_suffix('.wp.covar_frac'))
#         bins_file = str(param_file.with_suffix('.wp.signal'))
#         deps = obs_files+(script,input_cov_matrix_file)
#         targets = [fisher_matrix_file]
    
#         cmd_string = "python %(script)s %(output)s %(input_cov)s --bins_file %(bins_file)s --parameter_file %(param_file)s --observable wp %(inputs)s" \
#                      % {"script": script,
#                         "output": fisher_matrix_file,
#                         "input_cov": input_cov_matrix_file,
#                         "bins_file": bins_file,
#                         "param_file": param_file,
#                         "inputs": ' '.join(input_string)}

#         yield {
#             'actions': [cmd_string],
#             'file_dep': deps,
#             'task_dep': ['compute_derivative_wp'],
#             'targets': targets,
#             'name': param_file
#         }

# def task_compute_fisher_matrix_ngal_frac():
#     """compute Fisher matrix for n_gal"""
#     script = "./Analysis/compute_fisher_matrix.py"
#     param_files = Path('./Covariances_0.1hMpc').glob('*.param_frac')

#     for param_file in param_files:
#         varied_params = get_varied_params_from_file(param_file)
#         param_files_list = compute_param_files_list(varied_params,all_param_files)
#         derivative_files = [('./Derivatives_0.1hMpc/ln_ngal_%s.txt' % param, param) for param_files, param in param_files_list]
#         input_string = ["-f %s %s" % file_tuple for file_tuple in derivative_files]

#         obs_files, params = zip(*derivative_files)
#         fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".ngal.fisher_frac"
#         input_cov_matrix_file = str(param_file.with_suffix('.ngal.covar_frac'))
#         deps = obs_files+(script,input_cov_matrix_file)
#         targets = [fisher_matrix_file]
    
#         cmd_string = "python %(script)s %(output)s %(input_cov)s %(inputs)s" \
#                      % {"script": script,
#                         "output": fisher_matrix_file,
#                         "input_cov": input_cov_matrix_file,
#                         "inputs": ' '.join(input_string)}

#         yield {
#             'actions': [cmd_string],
#             'file_dep': deps,
#             'targets': targets,
#             'name': param_file
#         }

## add Fisher matrices

def task_add_fisher_matrices():
    """add Fisher matrices of w_p and \Delta\Sigma"""
    script = './Analysis/add_fisher_matrices.py'
    param_files = Path('./Covariances_0.1hMpc').glob('*.param')

    for param_file in param_files:
        wp_fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".wp.fisher"
        ds_fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".DeltaSigma.fisher"
        ngal_fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".ngal.fisher"
        output_fisher = './Forecasts_0.1hMpc/' + param_file.name + '.fisher'
        output_cov = './Forecasts_0.1hMpc/' + param_file.name + '.parameter_covariance'

        fisher_files = [wp_fisher_matrix_file, ds_fisher_matrix_file, ngal_fisher_matrix_file]
        deps = fisher_files+[script]
        targets = [output_fisher, output_cov]
    
        yield {
            'actions': ["python %(script)s %(output_fisher)s %(output_cov)s %(inputs)s"
                        % {"script": script,
                           "output_fisher": output_fisher,
                           "output_cov": output_cov,
                           "inputs": ' '.join(fisher_files)}],
            'file_dep': deps,
            'task_dep': ['compute_fisher_matrix_DeltaSigma','compute_fisher_matrix_wp',
                         'compute_fisher_matrix_ngal'],
            'targets': targets,
            'name': param_file
        }

def task_add_fisher_matrices_centralsonly():
    """add Fisher matrices of w_p and \Delta\Sigma"""
    script = './Analysis/add_fisher_matrices.py'
    param_files = Path('./Covariances_0.1hMpc').glob('*.param')

    for param_file in param_files:
        wp_fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".centralsonly.wp.fisher"
        ds_fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".centralsonly.DeltaSigma.fisher"
        ngal_fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".centralsonly.ngal.fisher"
        output_fisher = './Forecasts_0.1hMpc/' + param_file.name + '.centralsonly.fisher'
        output_cov = './Forecasts_0.1hMpc/' + param_file.name + '.centralsonly.parameter_covariance'

        fisher_files = [wp_fisher_matrix_file, ds_fisher_matrix_file, ngal_fisher_matrix_file]
        deps = fisher_files+[script]
        targets = [output_fisher, output_cov]
    
        yield {
            'actions': ["python %(script)s %(output_fisher)s %(output_cov)s %(inputs)s"
                        % {"script": script,
                           "output_fisher": output_fisher,
                           "output_cov": output_cov,
                           "inputs": ' '.join(fisher_files)}],
            'file_dep': deps,
            'task_dep': ['compute_fisher_matrix_DeltaSigma_centralsonly','compute_fisher_matrix_wp_centralsonly',
                         'compute_fisher_matrix_ngal_centralsonly'],
            'targets': targets,
            'name': param_file
        }

def task_add_fisher_matrices_DeltaSigma_ngal():
    """add Fisher matrices of \Delta\Sigma and ngal"""
    script = './Analysis/add_fisher_matrices.py'
    param_files = Path('./Covariances_0.1hMpc').glob('*.param')

    for param_file in param_files:
        ds_fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".DeltaSigma.fisher"
        ngal_fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".ngal.fisher"
        output_fisher = './Forecasts_0.1hMpc/' + param_file.name + '.DS+ngal.fisher'
        output_cov = './Forecasts_0.1hMpc/' + param_file.name + '.DS+ngal.parameter_covariance'

        fisher_files = [ds_fisher_matrix_file, ngal_fisher_matrix_file]
        deps = fisher_files+[script]
        targets = [output_fisher, output_cov]
    
        yield {
            'actions': ["python %(script)s %(output_fisher)s %(output_cov)s %(inputs)s"
                        % {"script": script,
                           "output_fisher": output_fisher,
                           "output_cov": output_cov,
                           "inputs": ' '.join(fisher_files)}],
            'file_dep': deps,
            'task_dep': ['compute_fisher_matrix_DeltaSigma',
                         'compute_fisher_matrix_ngal'],
            'targets': targets,
            'name': param_file
        }

def task_add_fisher_matrices_wp_ngal():
    """add Fisher matrices of w_p and ngal"""
    script = './Analysis/add_fisher_matrices.py'
    param_files = Path('./Covariances_0.1hMpc').glob('*.param')

    for param_file in param_files:
        wp_fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".wp.fisher"
        ngal_fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".ngal.fisher"

        output_fisher = './Forecasts_0.1hMpc/' + param_file.name + '.wp+ngal.fisher'
        output_cov = './Forecasts_0.1hMpc/' + param_file.name + '.wp+ngal.parameter_covariance'

        fisher_files = [wp_fisher_matrix_file, ngal_fisher_matrix_file]
        deps = fisher_files+[script]
        targets = [output_fisher, output_cov]
    
        yield {
            'actions': ["python %(script)s %(output_fisher)s %(output_cov)s %(inputs)s"
                        % {"script": script,
                           "output_fisher": output_fisher,
                           "output_cov": output_cov,
                           "inputs": ' '.join(fisher_files)}],
            'file_dep': deps,
            'task_dep': ['compute_fisher_matrix_wp',
                         'compute_fisher_matrix_ngal'],
            'targets': targets,
            'name': param_file
        }

# def task_add_fisher_matrices_frac():
#     """add Fisher matrices of w_p and \Delta\Sigma"""
#     script = './Analysis/add_fisher_matrices.py'
#     param_files = Path('./Covariances_0.1hMpc').glob('*.param_frac')

#     for param_file in param_files:
#         wp_fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".wp.fisher_frac"
#         ds_fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".DeltaSigma.fisher_frac"
#         ngal_fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".ngal.fisher_frac"
#         output_fisher = './Forecasts_0.1hMpc/' + param_file.name + '.fisher_frac'
#         output_cov = './Forecasts_0.1hMpc/' + param_file.name + '.parameter_covariance_frac'

#         fisher_files = [wp_fisher_matrix_file, ds_fisher_matrix_file, ngal_fisher_matrix_file]
#         deps = fisher_files+[script]
#         targets = [output_fisher, output_cov]
    
#         yield {
#             'actions': ["python %(script)s %(output_fisher)s %(output_cov)s %(inputs)s"
#                         % {"script": script,
#                            "output_fisher": output_fisher,
#                            "output_cov": output_cov,
#                            "inputs": ' '.join(fisher_files)}],
#             'file_dep': deps,
#             'task_dep': ['compute_fisher_matrix_DeltaSigma_frac','compute_fisher_matrix_wp_frac',
#                          'compute_fisher_matrix_ngal'],
#             'targets': targets,
#             'name': param_file
#         }

# def task_add_fisher_matrices_DeltaSigma_ngal_frac():
#     """add Fisher matrices of \Delta\Sigma and ngal"""
#     script = './Analysis/add_fisher_matrices.py'
#     param_files = Path('./Covariances_0.1hMpc').glob('*.param_frac')

#     for param_file in param_files:
#         ds_fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".DeltaSigma.fisher_frac"
#         ngal_fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".ngal.fisher_frac"
#         output_fisher = './Forecasts_0.1hMpc/' + param_file.name + '.DS+ngal.fisher_frac'
#         output_cov = './Forecasts_0.1hMpc/' + param_file.name + '.DS+ngal.parameter_covariance_frac'

#         fisher_files = [ds_fisher_matrix_file, ngal_fisher_matrix_file]
#         deps = fisher_files+[script]
#         targets = [output_fisher, output_cov]
    
#         yield {
#             'actions': ["python %(script)s %(output_fisher)s %(output_cov)s %(inputs)s"
#                         % {"script": script,
#                            "output_fisher": output_fisher,
#                            "output_cov": output_cov,
#                            "inputs": ' '.join(fisher_files)}],
#             'file_dep': deps,
#             'task_dep': ['compute_fisher_matrix_DeltaSigma_frac',
#                          'compute_fisher_matrix_ngal'],
#             'targets': targets,
#             'name': param_file
#         }

# def task_add_fisher_matrices_wp_ngal_frac():
#     """add Fisher matrices of w_p and ngal"""
#     script = './Analysis/add_fisher_matrices.py'
#     param_files = Path('./Covariances_0.1hMpc').glob('*.param_frac')

#     for param_file in param_files:
#         wp_fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".wp.fisher_frac"
#         ngal_fisher_matrix_file = "./Forecasts_0.1hMpc/" + param_file.name + ".ngal.fisher_frac"

#         output_fisher = './Forecasts_0.1hMpc/' + param_file.name + '.wp+ngal.fisher_frac'
#         output_cov = './Forecasts_0.1hMpc/' + param_file.name + '.wp+ngal.parameter_covariance_frac'

#         fisher_files = [wp_fisher_matrix_file, ngal_fisher_matrix_file]
#         deps = fisher_files+[script]
#         targets = [output_fisher, output_cov]
        
#         yield {
#             'actions': ["python %(script)s %(output_fisher)s %(output_cov)s %(inputs)s"
#                         % {"script": script,
#                            "output_fisher": output_fisher,
#                            "output_cov": output_cov,
#                            "inputs": ' '.join(fisher_files)}],
#             'file_dep': deps,
#             'task_dep': ['compute_fisher_matrix_wp_frac',
#                          'compute_fisher_matrix_ngal'],
#             'targets': targets,
#             'name': param_file
#         }

## make ellipse plots
def task_plot_ellipses():
    script = './Plotting/print_parameter_covariance.py'
    param_files = Path('./Covariances_0.1hMpc').glob('*.param')

    for param_file in param_files:
        cov_file = './Forecasts_0.1hMpc/' + param_file.name + '.parameter_covariance'
        deps = [cov_file, script]
        targets = ['./Forecasts_0.1hMpc/' + param_file.name + '.ellipses.pdf']

        yield {
            'actions': ["python %(script)s %(output)s %(input)s"
                        % {"script": script,
                           "input": cov_file,
                           "output": targets[0],
                       }],
            'file_dep': deps,
            'task_dep': ['add_fisher_matrices'],
            'targets': targets,
            'name': param_file
        }

def task_plot_ellipses_centralsonly():
    script = './Plotting/print_parameter_covariance.py'
    param_files = Path('./Covariances_0.1hMpc').glob('*.param')

    for param_file in param_files:
        cov_file = './Forecasts_0.1hMpc/' + param_file.name + '.centralsonly.parameter_covariance'
        deps = [cov_file, script]
        targets = ['./Forecasts_0.1hMpc/' + param_file.name + '.centralsonly.ellipses.pdf']

        yield {
            'actions': ["python %(script)s %(output)s %(input)s"
                        % {"script": script,
                           "input": cov_file,
                           "output": targets[0],
                       }],
            'file_dep': deps,
            'task_dep': ['add_fisher_matrices'],
            'targets': targets,
            'name': param_file
        }

# def task_plot_ellipses_frac():
#     script = './Plotting/print_parameter_covariance.py'
#     param_files = Path('./Covariances_0.1hMpc').glob('*.param_frac')

#     for param_file in param_files:
#         cov_file = './Forecasts_0.1hMpc/' + param_file.name + '.parameter_covariance_frac'
#         deps = [cov_file, script]
#         targets = ['./Forecasts_0.1hMpc/' + param_file.name + '.ellipses_frac.pdf']

#         yield {
#             'actions': ["python %(script)s %(output)s %(input)s"
#                         % {"script": script,
#                            "input": cov_file,
#                            "output": targets[0],
#                        }],
#             'file_dep': deps,
#             'task_dep': ['add_fisher_matrices'],
#             'targets': targets,
#             'name': param_file
#         }

def task_plot_ellipses_wp_ngal():
    script = './Plotting/print_parameter_covariance.py'
    param_files = Path('./Covariances_0.1hMpc').glob('*.param')

    for param_file in param_files:
        cov_file = './Forecasts_0.1hMpc/' + param_file.name + '.wp+ngal.parameter_covariance'
        deps = [cov_file, script]
        targets = ['./Forecasts_0.1hMpc/' + param_file.name + '.wp+ngal.ellipses.pdf']

        yield {
            'actions': ["python %(script)s %(output)s %(input)s"
                        % {"script": script,
                           "input": cov_file,
                           "output": targets[0],
                       }],
            'file_dep': deps,
            'task_dep': ['add_fisher_matrices_wp_ngal'],
            'targets': targets,
            'name': param_file
        }

# def task_plot_ellipses_DeltaSigma_ngal():
#     script = './Plotting/print_parameter_covariance.py'
#     param_files = Path('./Covariances_0.1hMpc').glob('*.param')

#     for param_file in param_files:
#         cov_file = './Forecasts_0.1hMpc/' + param_file.name + '.DS+ngal.parameter_covariance'
#         deps = [cov_file, script]
#         targets = ['./Forecasts_0.1hMpc/' + param_file.name + '.DS+ngal.ellipses.pdf']

#         yield {
#             'actions': ["python %(script)s %(output)s %(input)s"
#                         % {"script": script,
#                            "input": cov_file,
#                            "output": targets[0],
#                        }],
#             'file_dep': deps,
#             'task_dep': ['add_fisher_matrices_DeltaSigma_ngal'],
#             'targets': targets,
#             'name': param_file
#         }

# def task_plot_ellipses_wp_ngal_frac():
#     script = './Plotting/print_parameter_covariance.py'
#     param_files = Path('./Covariances_0.1hMpc').glob('*.param_frac')

#     for param_file in param_files:
#         cov_file = './Forecasts_0.1hMpc/' + param_file.name + '.wp+ngal.parameter_covariance_frac'
#         deps = [cov_file, script]
#         targets = ['./Forecasts_0.1hMpc/' + param_file.name + '.wp+ngal.ellipses_frac.pdf']

#         yield {
#             'actions': ["python %(script)s %(output)s %(input)s"
#                         % {"script": script,
#                            "input": cov_file,
#                            "output": targets[0],
#                        }],
#             'file_dep': deps,
#             'task_dep': ['add_fisher_matrices_wp_ngal_frac'],
#             'targets': targets,
#             'name': param_file
#         }

# def task_plot_ellipses_DeltaSigma_ngal_frac():
#     script = './Plotting/print_parameter_covariance.py'
#     param_files = Path('./Covariances_0.1hMpc').glob('*.param_frac')

#     for param_file in param_files:
#         cov_file = './Forecasts_0.1hMpc/' + param_file.name + '.DS+ngal.parameter_covariance_frac'
#         deps = [cov_file, script]
#         targets = ['Forecasts/' + param_file.name + '.DS+ngal.ellipses_frac.pdf']

#         yield {
#             'actions': ["python %(script)s %(output)s %(input)s"
#                         % {"script": script,
#                            "input": cov_file,
#                            "output": targets[0],
#                        }],
#             'file_dep': deps,
#             'task_dep': ['add_fisher_matrices_DeltaSigma_ngal_frac'],
#             'targets': targets,
#             'name': param_file
#         }
