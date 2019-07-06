from pipeline_defs import *
from pathlib import Path
import configparser
import h5py as h5
import glob
import os

param_files = param_files_in_dir(param_dir)
param_files_centralsonly = param_files_in_dir_centralsonly(param_dir)

def task_create_bin_file():
    """create bin file for use in Corrfunc"""
    
    script = "./Analysis/create_bins_file.py"
    binsfile = binfile_this_sim(rmin, rmax, halo_working_directory)
    deps = [script]    
    targets = [binsfile]
    yield {
        'actions': ["python %(script)s %(rmin)s %(rmax)s %(nbins)s %(output_file)s"
                   % {"script": script, "rmin": rmin, "rmax": rmax, "nbins": nbins, "output_file": binsfile}],
        'file_dep': deps,
        'targets':targets,
        'name':binsfile,
    }

###
### compute results that do NOT depend on galaxy population ###
###

def task_compute_environmental_densities():
    """compute environmental densities within some specified range of each halo"""
    program = "./fastcorrelation/density"
    
    rmax = 8.0 # h^-1 Mpc top-hat
    logdM = 0.1 # mass function bin width (dex)

    for subdir in subdirectories:
        halo_file = hdf5_Rockstar_catalog_this_sim(subdir)
        FOF_subdir = str(subdir).replace("Rockstar", "FOF")
        subsample = hdf5_particles_subsample_this_sim(FOF_subdir)
        deps = [program, halo_file, subsample]
        targets = [hdf5_Rockstar_env_this_sim(subdir)]

        if Path(subsample).exists():
            yield {
                'actions': ["%(program)s %(rmax)s %(boxsize)s %(logdM)s %(halo_file)s %(particle_file)s %(output_file)s"
                            % {"program": program, 
                               "rmax": rmax,
                               "boxsize": boxsize,
                               "logdM": logdM,
                               "particle_file": subsample,
                               "halo_file": halo_file,
                               "output_file": targets[0]}],
                'file_dep': deps,
                'targets': targets,
                'name': halo_file,
            }

def task_compute_mass_functions():
    for subdir in subdirectories:
        header = header_file_this_sim(subdir)
        halos = hdf5_Rockstar_catalog_this_sim(subdir)
        script = "./Analysis/compute_mass_function.py"

        deps = [header, halos, script]
        targets = [mass_function_this_sim(subdir)]
        action = "python %(script)s %(header)s %(halos)s %(target)s" \
                 % {"script": script,
                    "header": header,
                    "target": targets[0],
                    "halos": halos,
                 }
        yield {
            'actions': [action],
            'file_dep': deps,
            'targets': targets,
            'name': str(subdir),
        }

def task_compute_matter_autocorrelation():
    """compute matter 2pcf using Corrfunc"""

    for subdir in subdirectories:
        FOF_subdir = str(subdir).replace("Rockstar", "FOF") #particle subsamples are in the corresponding FOF directory
        matter_auto = txt_matter_autocorrelation_this_sim(FOF_subdir)
        subsample = hdf5_particles_subsample_this_sim(FOF_subdir)
        script = "./Analysis/autocorrelation_Corrfunc.py"
        binsfile = binfile_this_sim(rmin, rmax, halo_working_directory)

        deps = [subsample, script, binsfile]
        targets = [matter_auto]
        
        if Path(subsample).exists():
            yield {
                'actions': ["python %(script)s --ignore_weights %(boxsize)s %(binsfile)s %(subsample)s %(matter_auto)s"
                            % {"script": script,
                               "boxsize": boxsize,
                               "binsfile": binsfile, "subsample": subsample, "matter_auto": matter_auto}, 
                            "cp %(matter_auto)s %(matter_auto_this_directory)s"
                            % {"matter_auto": matter_auto,
                               "matter_auto_this_directory": matter_auto.replace("FOF", "Rockstar")}],
                'file_dep': deps,
                'targets': targets,
                'name': FOF_subdir,
                }

def task_compute_linear_matter_autocorrelation():
    """compute matter 2pcf from the linear power spectrum."""

    for subdir in subdirectories:
        power_subdir = str(subdir).replace("Rockstar", "power") #particle subsamples are in the corresponding FOF directory
        power_spectrum = str(power_subdir / Path("../info/camb_matterpower.dat"))
        matter_auto = txt_linear_matter_autocorrelation_this_sim(subdir)
        header = header_file_this_sim(subdir)
        script = "./Analysis/compute_linear_matter_correlation.py"

        deps = [power_spectrum, script]
        targets = [matter_auto]
        
        yield {
            'actions': ["python %(script)s %(power_spectrum)s %(binmin)s %(binmax)s %(nbins)s %(header)s %(matter_auto)s"
                        % {"script": script,
                           "power_spectrum": power_spectrum,
                           "binmin": rmin,
                           "binmax": rmax,
                           "nbins": nbins,
                           "header": header,
                           "matter_auto": matter_auto}],
            'file_dep': deps,
            'targets': targets,
            'name': power_subdir,
        }

def task_compute_nonlinear_matter_bias():
    """compute nonlinear matter 'bias' from the outputs of fastcorrelation."""

    for subdir in subdirectories:
        matter_file = txt_matter_autocorrelation_this_sim(subdir)
        lin_matter_file = txt_linear_matter_autocorrelation_this_sim(subdir)
        script = "./Analysis/compute_bias.py"
        
        deps = [lin_matter_file, matter_file, script]
        targets = [txt_nonlinear_matter_bias_this_sim(subdir)]

        yield {
            'actions': ["python %(script)s %(target)s %(matter_file)s %(lin_matter_file)s"
                        % {"script": script, "matter_file": matter_file, "target": targets[0], "lin_matter_file": lin_matter_file}],
            'file_dep': deps,
            'task_dep': ['compute_matter_autocorrelation', 'compute_linear_matter_autocorrelation'],
            'targets': targets,
            'name': str(matter_file),
        }

def task_compute_ln_bnl():
    """compute ln b_nl"""

    for subdir in subdirectories:
        correlation_file = txt_nonlinear_matter_bias_this_sim(subdir)
        script = "./Analysis/compute_ln_function.py"
        deps = [correlation_file, script]
        targets = [txt_ln_bnl_this_sim(subdir)]
            
        yield {
            'actions': ["python %(script)s %(correlation_file)s %(output)s"
                        % {"script": script,
                           "correlation_file": correlation_file,
                           "output": targets[0]}],
            'file_dep': deps,
            'targets': targets,
            'name': str(correlation_file),
        }

###
### compute results that *do* depend on galaxy population
###

### compute mocks ###

def task_compute_HOD_mocks():
    """compute mocks using HOD parameters specified in files"""

    for param_file in param_files:
        subdir = subdir_from_param_file(param_file)
        header = header_file_this_sim(subdir)
        halos = hdf5_Rockstar_catalog_this_sim(subdir)
        env = hdf5_Rockstar_env_this_sim(subdir)
        massfun = mass_function_this_sim(subdir)
        script = "./Analysis/compute_mock.py"
        external_code = "./cHOD/compute_mocks"

        deps = [param_file, header, halos, env, massfun, script, external_code]
        targets = [hdf5_HOD_mock_this_param(param_file)]
        action = "python %(script)s %(param_file)s %(header)s %(halos)s %(target)s %(env)s %(massfun)s" \
                 % {"script": script,
                    "param_file": param_file,
                    "header": header, "target": targets[0], "halos": halos, "env": env,
                    "massfun": massfun,}

        yield {
            'actions': [action],
            'task_dep': ['compute_mass_functions'],
            'file_dep': deps,
            'targets': targets,
            'name': str(param_file),
        }

def task_compute_HOD_mocks_centralsonly():
    """compute mocks using HOD parameters specified in files"""

    for param_file in param_files_centralsonly:
        subdir = subdir_from_param_file(param_file)
        header = header_file_this_sim(subdir)
        halos = hdf5_Rockstar_catalog_this_sim(subdir)
        env = hdf5_Rockstar_env_this_sim(subdir)
        massfun = mass_function_this_sim(subdir)
        script = "./Analysis/compute_mock.py"
        external_code = "./cHOD/compute_mocks"

        deps = [param_file, header, halos, env, massfun, script, external_code]
        targets = [hdf5_HOD_mock_this_param_centralsonly(param_file)]
        action = "python %(script)s --centrals_only %(param_file)s %(header)s %(halos)s %(target)s %(env)s %(massfun)s" \
                 % {"script": script,
                    "param_file": param_file,
                    "header": header, "target": targets[0], "halos": halos, "env": env,
                    "massfun": massfun,}
        
        yield {
            'actions': [action],
            'task_dep': ['compute_mass_functions'],
            'file_dep': deps,
            'targets': targets,
            'name': str(param_file),
        }

### compute auto- and cross-correlation functions ###

def task_compute_galaxy_autocorrelation():
    """compute 2pcf of galaxies using Corrfunc."""

    for param_file in param_files:
        mock_file = hdf5_HOD_mock_this_param(param_file)
        script = "./Analysis/autocorrelation_Corrfunc.py"
        binsfile = binfile_this_sim(rmin, rmax, halo_working_directory)
        deps = [mock_file, script, binsfile]
        targets = [txt_galaxy_autocorrelation_this_param(param_file)]

        yield {
            'actions': ["python %(script)s %(boxsize)s %(binsfile)s %(mock_file)s %(target)s"
                        % {"script": script,
                           "boxsize": boxsize, "binsfile": binsfile, "mock_file": mock_file, "target": targets[0]}],
            'file_dep': deps,
            'task_dep': ['compute_HOD_mocks'],
            'targets': targets,
            'name': str(mock_file),
        }

def task_compute_galaxy_autocorrelation_centralsonly():
    """compute 2pcf of galaxies using Corrfunc."""

    for param_file in param_files_centralsonly:
        mock_file = hdf5_HOD_mock_this_param_centralsonly(param_file)
        script = "./Analysis/autocorrelation_Corrfunc.py"
        binsfile = binfile_this_sim(rmin, rmax, halo_working_directory)
        deps = [mock_file, script, binsfile]
        targets = [txt_galaxy_autocorrelation_this_param_centralsonly(param_file)]

        yield {
            'actions': ["python %(script)s %(boxsize)s %(binsfile)s %(mock_file)s %(target)s --centrals_only"
                        % {"script": script,
                           "boxsize": boxsize, "binsfile": binsfile, "mock_file": mock_file, "target": targets[0]}],
            'file_dep': deps,
            'task_dep': ['compute_HOD_mocks'],
            'targets': targets,
            'name': str(mock_file),
        }

def task_compute_galaxy_matter_crosscorrelation():
    """computer galaxy-matter crosscorrelation function using Corrfunc"""
    
    for param_file in param_files:
        mock_file = hdf5_HOD_mock_this_param(param_file)
        subdir = subdir_from_param_file(param_file)
        FOF_subdir = str(subdir).replace("Rockstar", "FOF")
        subsample = hdf5_particles_subsample_this_sim(FOF_subdir)
        script = "./Analysis/crosscorrelation_Corrfunc.py"
        binsfile = binfile_this_sim(rmin, rmax, halo_working_directory)
        deps = [mock_file, subsample, script, binsfile]
        targets = [txt_galaxy_matter_crosscorrelation_this_param(param_file)]
        yield {
            'actions': ["python %(script)s %(boxsize)s %(binsfile)s %(mock_file)s %(subsample)s %(target)s"
                        % {"script": script,
                           "boxsize": boxsize,
                           "binsfile": binsfile, "mock_file": mock_file, "target": targets[0], "subsample": subsample}],
            'file_dep': deps,
            'task_dep': ['compute_HOD_mocks'],
            'targets': targets,
            'name': str(mock_file),
        }

def task_compute_galaxy_matter_crosscorrelation_centralsonly():
    """computer galaxy-matter crosscorrelation function using Corrfunc"""
    
    for param_file in param_files_centralsonly:
        mock_file = hdf5_HOD_mock_this_param_centralsonly(param_file)
        subdir = subdir_from_param_file(param_file)
        FOF_subdir = str(subdir).replace("Rockstar", "FOF")
        subsample = hdf5_particles_subsample_this_sim(FOF_subdir)
        script = "./Analysis/crosscorrelation_Corrfunc.py"
        binsfile = binfile_this_sim(rmin, rmax, halo_working_directory)
        deps = [mock_file, subsample, script, binsfile]
        targets = [txt_galaxy_matter_crosscorrelation_this_param_centralsonly(param_file)]
        yield {
            'actions': ["python %(script)s %(boxsize)s %(binsfile)s %(mock_file)s %(subsample)s %(target)s --centrals_only" 
                        % {"script": script,
                           "boxsize": boxsize,
                           "binsfile": binsfile, "mock_file": mock_file, "target": targets[0], "subsample": subsample}],
            'file_dep': deps,
            'task_dep': ['compute_HOD_mocks'],
            'targets': targets,
            'name': str(mock_file),
        }
                
### compute derived quantities from the averaged galaxy correlation functions ###

def task_compute_galaxy_bias():
    """compute galaxy bias from the outputs of fastcorrelation."""

    for param_file in param_files:
        galaxy_file = txt_galaxy_autocorrelation_this_param(param_file)
        subdir = subdir_from_param_file(param_file)
        matter_file = txt_matter_autocorrelation_this_sim(subdir)
        script = "./Analysis/compute_bias.py"
        deps = [galaxy_file, matter_file, script]
        targets = [txt_galaxy_bias_this_param(param_file)]
            
        yield {
            'actions': ["python %(script)s %(target)s %(galaxy_file)s %(matter_file)s --regularize"
                        % {"script": script, "galaxy_file": galaxy_file, "target": targets[0], "matter_file": matter_file}],
            'file_dep': deps,
            'task_dep': ['compute_galaxy_autocorrelation', 'compute_matter_autocorrelation'],
            'targets': targets,
            'name': str(galaxy_file),
        }

def task_compute_galaxy_matter_correlation_coefficient():
    """compute galaxy-matter correlation coefficient from the outputs of fastcorrelation."""
    
    for param_file in param_files:
        galaxy_file = txt_galaxy_autocorrelation_this_param(param_file)
        galaxy_matter_file = txt_galaxy_matter_crosscorrelation_this_param(param_file)
        subdir = subdir_from_param_file(param_file)
        matter_file = txt_matter_autocorrelation_this_sim(subdir)
        script = "./Analysis/compute_correlation_coefficient.py"
        deps = [galaxy_file, matter_file, galaxy_matter_file, script]
        targets = [txt_galaxy_matter_correlation_coefficient_this_param(param_file)]

        yield {
            'actions': ["python %(script)s %(target)s %(galaxy_file)s %(matter_file)s %(galaxy_matter_file)s"
                        % {"script": script,
                           "galaxy_file": galaxy_file,
                           "target": targets[0], "matter_file": matter_file, "galaxy_matter_file": galaxy_matter_file}],
            'file_dep': deps,
            'task_dep': ['compute_galaxy_autocorrelation', 'compute_matter_autocorrelation', 'compute_galaxy_matter_crosscorrelation'],
            'targets': targets,
            'name': str(galaxy_file)
        }

def task_compute_galaxy_bias_centralsonly():
    """compute galaxy bias from the outputs of fastcorrelation."""

    for param_file in param_files_centralsonly:
        galaxy_file = txt_galaxy_autocorrelation_this_param_centralsonly(param_file)
        subdir = subdir_from_param_file(param_file)
        matter_file = txt_matter_autocorrelation_this_sim(subdir)
        script = "./Analysis/compute_bias.py"
        deps = [galaxy_file, matter_file, script]
        targets = [txt_galaxy_bias_this_param_centralsonly(param_file)]
            
        yield {
            'actions': ["python %(script)s %(target)s %(galaxy_file)s %(matter_file)s --regularize"
                        % {"script": script, "galaxy_file": galaxy_file, "target": targets[0], "matter_file": matter_file}],
            'file_dep': deps,
            'task_dep': ['compute_galaxy_autocorrelation', 'compute_matter_autocorrelation'],
            'targets': targets,
            'name': str(galaxy_file),
        }

def task_compute_galaxy_matter_correlation_coefficient_centralsonly():
    """compute galaxy-matter correlation coefficient from the outputs of fastcorrelation."""
    
    for param_file in param_files_centralsonly:
        galaxy_file = txt_galaxy_autocorrelation_this_param_centralsonly(param_file)
        galaxy_matter_file = txt_galaxy_matter_crosscorrelation_this_param_centralsonly(param_file)
        subdir = subdir_from_param_file(param_file)
        matter_file = txt_matter_autocorrelation_this_sim(subdir)
        script = "./Analysis/compute_correlation_coefficient.py"
        deps = [galaxy_file, matter_file, galaxy_matter_file, script]
        targets = [txt_galaxy_matter_correlation_coefficient_this_param_centralsonly(param_file)]

        yield {
            'actions': ["python %(script)s %(target)s %(galaxy_file)s %(matter_file)s %(galaxy_matter_file)s"
                        % {"script": script,
                           "galaxy_file": galaxy_file,
                           "target": targets[0], "matter_file": matter_file, "galaxy_matter_file": galaxy_matter_file}],
            'file_dep': deps,
            'task_dep': ['compute_galaxy_autocorrelation', 'compute_matter_autocorrelation', 'compute_galaxy_matter_crosscorrelation'],
            'targets': targets,
            'name': str(galaxy_file)
        }

## compute ln b_nl, ln b_g, ln r_gm ##

def task_compute_ln_bg():
    """compute ln b_g"""

    for param_file in param_files:
        correlation_file = txt_galaxy_bias_this_param(param_file)
        script = "./Analysis/compute_ln_function.py"
        deps = [correlation_file, script]
        targets = [txt_ln_bg_this_param(param_file)]
            
        yield {
            'actions': ["python %(script)s %(correlation_file)s %(output)s"
                        % {"script": script,
                           "correlation_file": correlation_file,
                           "output": targets[0]}],
            'file_dep': deps,
            'targets': targets,
            'name': str(correlation_file),
        }

def task_compute_ln_bg_centralsonly():
    """compute ln b_g"""

    for param_file in param_files_centralsonly:
        correlation_file = txt_galaxy_bias_this_param_centralsonly(param_file)
        script = "./Analysis/compute_ln_function.py"
        deps = [correlation_file, script]
        targets = [txt_ln_bg_this_param_centralsonly(param_file)]
            
        yield {
            'actions': ["python %(script)s %(correlation_file)s %(output)s"
                        % {"script": script,
                           "correlation_file": correlation_file,
                           "output": targets[0]}],
            'file_dep': deps,
            'targets': targets,
            'name': str(correlation_file),
        }

def task_compute_ln_rgm():
    """compute ln r_gm"""

    for param_file in param_files:
        correlation_file = txt_galaxy_matter_correlation_coefficient_this_param(param_file)
        script = "./Analysis/compute_ln_function.py"        
        deps = [correlation_file, script]
        targets = [txt_ln_rgm_this_param(param_file)]

        yield {
            'actions': ["python %(script)s %(correlation_file)s %(output)s"
                        % {"script": script,
                           "correlation_file": correlation_file,
                           "output": targets[0]}],
            'file_dep': deps,
            'targets': targets,
            'name': str(correlation_file),
        }

def task_compute_ln_rgm_centralsonly():
    """compute ln r_gm"""

    for param_file in param_files_centralsonly:
        correlation_file = txt_galaxy_matter_correlation_coefficient_this_param_centralsonly(param_file)
        script = "./Analysis/compute_ln_function.py"
        deps = [correlation_file, script]
        targets = [txt_ln_rgm_this_param_centralsonly(param_file)]
            
        yield {
            'actions': ["python %(script)s %(correlation_file)s %(output)s"
                        % {"script": script,
                           "correlation_file": correlation_file,
                           "output": targets[0]}],
            'file_dep': deps,
            'targets': targets,
            'name': str(correlation_file),
        }

## compute derived quantites from the matter correlation function ##

def task_reconstruct_xi_gg():
    """compute galaxy correlation from b_g, b_m, mm_lin."""

    for param_file in param_files:
        galaxy_bias_file = txt_galaxy_bias_this_param(param_file)
        subdir = subdir_from_param_file(param_file)
        matter_file = txt_linear_matter_autocorrelation_this_sim(subdir)
        matter_bias_file = txt_nonlinear_matter_bias_this_sim(subdir)
        script = "./Analysis/reconstruct_xi_gg.py"
        deps = [galaxy_bias_file, matter_file, matter_bias_file, script]
        targets = [txt_reconstructed_xi_gg_this_param(param_file)]
        
        yield {
            'actions': ["python %(script)s %(target)s %(galaxy_bias_file)s %(matter_bias_file)s %(linear_matter_file)s"
                        % {"script": script,
                           "galaxy_bias_file": galaxy_bias_file,
                           "target": targets[0],
                           "matter_bias_file": matter_bias_file,
                           "linear_matter_file": matter_file}],
            'file_dep': deps,
            'task_dep': ['compute_galaxy_bias', 'compute_nonlinear_matter_bias', 'compute_linear_matter_autocorrelation'],
            'targets': targets,
            'name': str(galaxy_bias_file),
        }

def task_reconstruct_xi_gm():
    """compute galaxy-matter correlation from b_g, b_m, r_gm, mm_lin."""

    for param_file in param_files:
        galaxy_bias_file = txt_galaxy_bias_this_param(param_file)
        r_gm_file = txt_galaxy_matter_correlation_coefficient_this_param(param_file)
        subdir = subdir_from_param_file(param_file)
        matter_file = txt_linear_matter_autocorrelation_this_sim(subdir)
        matter_bias_file = txt_nonlinear_matter_bias_this_sim(subdir)
        script = "./Analysis/reconstruct_xi_gm.py"        
        deps = [galaxy_bias_file, matter_file, r_gm_file, matter_bias_file, script]
        targets = [txt_reconstructed_xi_gm_this_param(param_file)]
            
        yield {
            'actions': ["python %(script)s %(target)s %(galaxy_bias_file)s %(matter_bias_file)s %(r_gm_file)s %(linear_matter_file)s"
                        % {"script": script,
                           "galaxy_bias_file": galaxy_bias_file,
                           "target": targets[0],
                           "matter_bias_file": matter_bias_file,
                           "r_gm_file": r_gm_file,
                           "linear_matter_file": matter_file}],
            'file_dep': deps,
            'task_dep': ['compute_galaxy_bias', 'compute_nonlinear_matter_bias', 'compute_linear_matter_autocorrelation', 'compute_galaxy_matter_correlation_coefficient'],
            'targets': targets,
            'name': str(galaxy_bias_file),
        }

def task_reconstruct_xi_gg_centralsonly():
    """compute galaxy correlation from b_g, b_m, mm_lin."""

    for param_file in param_files_centralsonly:
        galaxy_bias_file = txt_galaxy_bias_this_param_centralsonly(param_file)
        subdir = subdir_from_param_file(param_file)
        matter_file = txt_linear_matter_autocorrelation_this_sim(subdir)
        matter_bias_file = txt_nonlinear_matter_bias_this_sim(subdir)
        script = "./Analysis/reconstruct_xi_gg.py"        
        deps = [galaxy_bias_file, matter_file, matter_bias_file, script]
        targets = [txt_reconstructed_xi_gg_this_param_centralsonly(param_file)]
            
        yield {
            'actions': ["python %(script)s %(target)s %(galaxy_bias_file)s %(matter_bias_file)s %(linear_matter_file)s"
                        % {"script": script,
                           "galaxy_bias_file": galaxy_bias_file,
                           "target": targets[0],
                           "matter_bias_file": matter_bias_file,
                           "linear_matter_file": matter_file}],
            'file_dep': deps,
            'task_dep': ['compute_galaxy_bias', 'compute_nonlinear_matter_bias', 'compute_linear_matter_autocorrelation'],
            'targets': targets,
            'name': str(galaxy_bias_file),
        }

def task_reconstruct_xi_gm():
    """compute galaxy-matter correlation from b_g, b_m, r_gm, mm_lin."""

    for param_file in param_files:
        galaxy_bias_file = txt_galaxy_bias_this_param(param_file)
        r_gm_file = txt_galaxy_matter_correlation_coefficient_this_param(param_file)
        subdir = subdir_from_param_file(param_file)
        matter_file = txt_linear_matter_autocorrelation_this_sim(subdir)
        matter_bias_file = txt_nonlinear_matter_bias_this_sim(subdir)
        script = "./Analysis/reconstruct_xi_gm.py"        
        deps = [galaxy_bias_file, matter_file, r_gm_file, matter_bias_file, script]
        targets = [txt_reconstructed_xi_gm_this_param(param_file)]
            
        yield {
            'actions': ["python %(script)s %(target)s %(galaxy_bias_file)s %(matter_bias_file)s %(r_gm_file)s %(linear_matter_file)s"
                        % {"script": script,
                           "galaxy_bias_file": galaxy_bias_file,
                           "target": targets[0],
                           "matter_bias_file": matter_bias_file,
                           "r_gm_file": r_gm_file,
                           "linear_matter_file": matter_file}],
            'file_dep': deps,
            'task_dep': ['compute_galaxy_bias', 'compute_nonlinear_matter_bias', 'compute_linear_matter_autocorrelation', 'compute_galaxy_matter_correlation_coefficient'],
            'targets': targets,
            'name': str(galaxy_bias_file),
        }

def task_reconstruct_xi_gm_centralsonly():
    """compute galaxy-matter correlation from b_g, b_m, r_gm, mm_lin."""

    for param_file in param_files_centralsonly:
        galaxy_bias_file = txt_galaxy_bias_this_param_centralsonly(param_file)
        r_gm_file = txt_galaxy_matter_correlation_coefficient_this_param_centralsonly(param_file)
        subdir = subdir_from_param_file(param_file)
        matter_file = txt_linear_matter_autocorrelation_this_sim(subdir)
        matter_bias_file = txt_nonlinear_matter_bias_this_sim(subdir)
        script = "./Analysis/reconstruct_xi_gm.py"        
        deps = [galaxy_bias_file, matter_file, r_gm_file, matter_bias_file, script]
        targets = [txt_reconstructed_xi_gm_this_param_centralsonly(param_file)]
            
        yield {
            'actions': ["python %(script)s %(target)s %(galaxy_bias_file)s %(matter_bias_file)s %(r_gm_file)s %(linear_matter_file)s"
                        % {"script": script,
                           "galaxy_bias_file": galaxy_bias_file,
                           "target": targets[0],
                           "matter_bias_file": matter_bias_file,
                           "r_gm_file": r_gm_file,
                           "linear_matter_file": matter_file}],
            'file_dep': deps,
            'task_dep': ['compute_galaxy_bias', 'compute_nonlinear_matter_bias', 'compute_linear_matter_autocorrelation', 'compute_galaxy_matter_correlation_coefficient'],
            'targets': targets,
            'name': str(galaxy_bias_file),
        }



