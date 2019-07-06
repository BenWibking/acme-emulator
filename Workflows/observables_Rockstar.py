from pipeline_defs import *
from pathlib import Path
import shlex
import configparser
from itertools import chain

"""
N.B.: For some reason doit runs tasks whose 'file_dep's do not exist. This is not what we want...
Workaround: Use 'task_dep' to additionally specify the tasks that create the 'file_dep's.
"""

#### compute observables

## HOD param files
param_files = param_files_in_dir(param_dir)

def observable_path(path, basepath="./Observables"):
    """change Path 'path' to be basepath / path.name"""
    return path
    # return str(Path(basepath) / Path(path).name)

## compute observables from reconstructed xi_gg, xi_gm ##

def task_smooth_bias():
    for param_file in param_files:
        correlation_file = txt_galaxy_bias_this_param(param_file)
        subdir = subdir_from_param_file(param_file)
        script = "./Analysis/smooth_function.py"
        deps = [correlation_file, script]
        targets = [observable_path(txt_smoothed_bias_this_param(param_file))]
        
        yield {
            'actions': ["python %(script)s %(correlation_file)s %(output)s"
                        % {"script": script,
                           "correlation_file": correlation_file,
                           "output": targets[0]}],
            'file_dep': deps,
            'targets': targets,
            'name': str(correlation_file),
        }

def task_compute_wp():
    """compute Omega_M * xsi_gm (for now)"""

    for param_file in param_files:
        correlation_file = txt_galaxy_autocorrelation_this_param(param_file)
        subdir = subdir_from_param_file(param_file)
        header_file = header_file_this_sim(subdir)
        script = "./Analysis/compute_wp.py"
        deps = [correlation_file, header_file, script]
        targets = [observable_path(txt_reconstructed_wp_this_param(param_file))]
            
        yield {
            'actions': ["python %(script)s %(correlation_file)s %(header_file)s %(output)s"
                        % {"script": script,
                           "correlation_file": correlation_file,
                           "header_file": header_file,
                           "output": targets[0]}],
            'file_dep': deps,
            'targets': targets,
            'name': str(correlation_file),
        }

def task_compute_delta_sigma():
    """compute DeltaSigma(r_p) from reconstructed xi_gm"""

    for param_file in param_files:
        correlation_file = txt_galaxy_matter_crosscorrelation_this_param(param_file)
        subdir = subdir_from_param_file(param_file)
        header_file = header_file_this_sim(subdir)
        script = "./Analysis/compute_DeltaSigma.py"
        deps = [correlation_file, header_file, script]
        targets = [observable_path(txt_reconstructed_DeltaSigma_this_param(param_file))]
            
        yield {
            'actions': ["python %(script)s %(correlation_file)s %(header_file)s %(output)s"
                        % {"script": script,
                           "correlation_file": correlation_file,
                           "output": targets[0], "header_file": header_file}],
            'file_dep': deps,
            'targets': targets,
            'name': str(correlation_file),
        }

def task_compute_wp_weighted():
    """compute Omega_M * xsi_gm (for now)"""

    for param_file in param_files:
        correlation_file = txt_galaxy_autocorrelation_weighted_this_param(param_file)
        subdir = subdir_from_param_file(param_file)
        header_file = header_file_this_sim(subdir)
        script = "./Analysis/compute_wp.py"
        deps = [correlation_file, header_file, script]
        targets = [observable_path(txt_wp_weighted_this_param(param_file))]
            
        yield {
            'actions': ["python %(script)s %(correlation_file)s %(header_file)s %(output)s"
                        % {"script": script,
                           "correlation_file": correlation_file,
                           "header_file": header_file,
                           "output": targets[0]}],
            'file_dep': deps,
            'targets': targets,
            'name': str(correlation_file),
        }

def task_compute_smoothed_bias_weighted():
    """compute Omega_M * xsi_gm (for now)"""

    for param_file in param_files:
        correlation_file = txt_galaxy_bias_weighted_this_param(param_file)
        subdir = subdir_from_param_file(param_file)
        script = "./Analysis/smooth_function.py"
        deps = [correlation_file, script]
        targets = [observable_path(txt_smoothed_bias_weighted_this_param(param_file))]
            
        yield {
            'actions': ["python %(script)s %(correlation_file)s %(output)s"
                        % {"script": script,
                           "correlation_file": correlation_file,
                           "output": targets[0]}],
            'file_dep': deps,
            'targets': targets,
            'name': str(correlation_file),
        }

def task_compute_delta_sigma_weighted():
    """compute DeltaSigma(r_p) from reconstructed xi_gm"""

    for param_file in param_files:
        correlation_file = txt_galaxy_matter_crosscorrelation_weighted_this_param(param_file)
        subdir = subdir_from_param_file(param_file)
        header_file = header_file_this_sim(subdir)
        script = "./Analysis/compute_DeltaSigma.py"
        deps = [correlation_file, header_file, script]
        targets = [observable_path(txt_DeltaSigma_weighted_this_param(param_file))]
            
        yield {
            'actions': ["python %(script)s %(correlation_file)s %(header_file)s %(output)s"
                        % {"script": script,
                           "correlation_file": correlation_file,
                           "output": targets[0], "header_file": header_file}],
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

## compute 'compensated' observables

def task_compute_DeltaSigma_baldauf():
    """compute DeltaSigma with small-scale info removed"""

    for param_file in param_files:
        correlation_file = observable_path(txt_reconstructed_DeltaSigma_this_param(param_file))
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

## compute log observables

def task_compute_log_wp():
    """compute log w_p"""

    for param_file in param_files:
        correlation_file = observable_path(txt_reconstructed_wp_this_param(param_file))
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
        correlation_file= observable_path(txt_reconstructed_DeltaSigma_this_param(param_file))
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


## compute log compensated observables

def task_compute_log_DeltaSigma_baldauf():
    """compute log DeltaSigma (Baldauf estimator) """

    for param_file in param_files:
        correlation_file = observable_path(txt_reconstructed_DeltaSigma_baldauf_this_param(param_file))
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

## compute log number density

def task_compute_log_number_density():
    """compute log number density"""

    for param_file in param_files:
        correlation_file = observable_path(txt_galaxy_number_density_this_param(param_file))
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

