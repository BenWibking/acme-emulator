from pipeline_defs import *
from pathlib import Path
import shlex
import configparser
from itertools import chain

from fisher_Rockstar import param_files_with_parameter_variation, get_varied_params_from_file, compute_param_files_list, is_log_param, deriv_cmd_string

all_varied_parameters = ['ngal','siglogM','M1_over_Mmin','alpha','q_env','del_gamma','f_cen','Omega_M','sigma_8']

use_log_parameter = ['ngal','siglogM','M1_over_Mmin','alpha','Omega_M','sigma_8','f_cen']
cosmological_parameters = ['sigma_8', 'Omega_M']
param_dir_alias = {}
param_dir_alias['Omega_M'] = 'H0'
param_dir_alias['sigma_8'] = 'sigma_8'
param_dir_alias['H0'] = 'H0'
sub_sim_dirs = ['fiducial', 'H0=64.26', 'H0=70.26', 'sigma_8=0.78', 'sigma_8=0.88']

sim_dirs = [halo_working_directory / subdir / Path(redshift) for subdir in sub_sim_dirs]
all_param_files = param_files_in_dir(param_dir)

param_files_list = compute_param_files_list(all_varied_parameters, all_param_files)
cosmo_param_files_list = compute_param_files_list(cosmological_parameters, all_param_files)

## copy fiducial ln b_g, ln b_nl, ln r_gm

def task_copy_fiducial_ln_bg():
    correlation_file = txt_ln_bg_this_param(Path(param_dir) / 'NHOD_fiducial.template_param')
    output = './Emulator/fiducial_ln_bg.txt'
    deps = [correlation_file]
    targets = [output]

    yield {
        'actions': ["cp %(input)s %(output)s"
                    % {"input": correlation_file, "output": output}],
        'file_dep': deps,
        'targets': targets,
        'name': working_directory
    }

def task_copy_fiducial_ln_rgm():
    correlation_file = txt_ln_rgm_this_param(Path(param_dir) / 'NHOD_fiducial.template_param')
    output = './Emulator/fiducial_ln_rgm.txt'
    deps = [correlation_file]
    targets = [output]

    yield {
        'actions': ["cp %(input)s %(output)s"
                    % {"input": correlation_file, "output": output}],
        'file_dep': deps,
        'targets': targets,
        'name': working_directory
    }

def task_copy_fiducial_ln_bnl():
    subdir =  halo_working_directory / 'fiducial' / redshift # 'fiducial' subdir
    correlation_file = txt_ln_bnl_this_sim(subdir)
    output = './Emulator/fiducial_ln_bnl.txt'
    deps = [correlation_file]
    targets = [output]

    yield {
        'actions': ["cp %(input)s %(output)s"
                    % {"input": correlation_file, "output": output}],
        'file_dep': deps,
        'targets': targets,
        'name': working_directory
    }

## compute derivatives of ln b_g, ln b_nl, ln r_gm

def task_compute_derivative_ln_bg():
    """compute derivative of ln b_g"""
    for param_files, param in param_files_list:
        correlation_files = [txt_ln_bg_this_param(x) for x in param_files]
        input_string = deriv_cmd_string(param_files, correlation_files, param)
        script = "./Analysis/compute_partial_derivative.py"
        deps = correlation_files+[script]
        output = './Emulator/ln_bg_%s.txt' % param
        targets = [output]
        
        yield {
            'actions': ["python %(script)s %(output)s %(inputs)s"
                        % {"script": script, "output": output, "inputs": ' '.join(input_string)}],
            'file_dep': deps,
            'targets': targets,
            'name': working_directory / param
        }

def task_compute_derivative_ln_rgm():
    """compute derivative of ln r_gm"""
    for param_files, param in param_files_list:
        correlation_files = [txt_ln_rgm_this_param(x) for x in param_files]
        input_string = deriv_cmd_string(param_files, correlation_files, param)
        script = "./Analysis/compute_partial_derivative.py"
        deps = correlation_files+[script]
        output = './Emulator/ln_rgm_%s.txt' % param
        targets = [output]
        
        yield {
            'actions': ["python %(script)s %(output)s %(inputs)s"
                        % {"script": script, "output": output, "inputs": ' '.join(input_string)}],
            'file_dep': deps,
            'targets': targets,
            'name': working_directory / param
        }

def task_compute_derivative_ln_bnl():
    """compute derivative of ln b_nl"""

    def txt_ln_bnl_this_param(x):
        return txt_ln_bnl_this_sim(subdir_from_param_file(x))

    for param_files, param in cosmo_param_files_list:
        correlation_files = [txt_ln_bnl_this_param(x) for x in param_files]
        input_string = deriv_cmd_string(param_files, correlation_files, param)
        script = "./Analysis/compute_partial_derivative.py"
        deps = correlation_files+[script]
        output = './Emulator/ln_bnl_%s.txt' % param
        targets = [output]

        yield {
            'actions': ["python %(script)s %(output)s %(inputs)s"
                        % {"script": script, "output": output, "inputs": ' '.join(input_string)}],
            'file_dep': deps,
            'targets': targets,
            'name': working_directory / param
        }
