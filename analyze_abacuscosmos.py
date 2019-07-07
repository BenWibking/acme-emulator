#!/usr/bin/env python

from pipeline_defs import *
from pathlib import Path
import os
import configparser

param_files = param_files_in_dir(param_dir)


def task_create_bin_file():

	"""create bin file for use in Corrfunc"""
	
	script = "./Analysis/create_bins_file.py"
	binsfile = binfile_this_sim(rmin, rmax, halo_working_directory)
	
	deps = [script]    
	targets = [binsfile]
	
	yield {
		'actions': [f"python {script} {rmin} {rmax} {nbins} {binsfile}"],
		'file_dep': deps,
		'targets': targets,
		'name': binsfile,
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
				'actions': [f"{program} {rmax} {boxsize} {logdM} {halo_file} {subsample} {targets[0]}"],
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

		yield {
			'actions': [f"python {script} {header} {halos} {targets[0]}"],
			'file_dep': deps,
			'targets': targets,
			'name': str(subdir),
		}


def task_compute_matter_autocorrelation():

	"""compute matter 2pcf using Corrfunc"""

	for subdir in subdirectories:
	
		# particle subsamples are in the corresponding FOF directory
		FOF_subdir = str(subdir).replace("Rockstar", "FOF")
		
		matter_auto = txt_matter_autocorrelation_this_sim(FOF_subdir)
		subsample = hdf5_particles_subsample_this_sim(FOF_subdir)
		script = "./Analysis/autocorrelation_Corrfunc.py"
		binsfile = binfile_this_sim(rmin, rmax, halo_working_directory)

		deps = [subsample, script, binsfile]
		targets = [matter_auto]
		
		if Path(subsample).exists():
		
			yield {
				'actions': [f"python {script} --ignore_weights {boxsize} {binsfile} {subsample} {matter_auto}", f"cp {matter_auto} {matter_auto.replace('FOF','Rockstar')}"],
				'file_dep': deps,
				'targets': targets,
				'name': FOF_subdir,
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
		action = f"python {script} {param_file} {header} {halos} {targets[0]} {env} {massfun}"

		yield {
			'actions': [action],
			'task_dep': ['compute_mass_functions'],
			'file_dep': deps,
			'targets': targets,
			'name': str(param_file),
		}

def task_compute_HOD_logMmin():

	"""compute mocks using HOD parameters specified in files"""

	for param_file in param_files:
	
		subdir = subdir_from_param_file(param_file)
		header = header_file_this_sim(subdir)
		halos = hdf5_Rockstar_catalog_this_sim(subdir)
		massfun = mass_function_this_sim(subdir)
		script = "./Analysis/compute_logMmin.py"

		deps = [param_file, header, halos, massfun, script]
		targets = [hdf5_HOD_logMmin_this_param(param_file)]
		action = f"python {script} {param_file} {header} {halos} {massfun} {targets[0]}"

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
			'actions': [f"python {script} {boxsize} {binsfile} {mock_file} {targets[0]}"],
			'file_dep': deps,
			'task_dep': ['compute_HOD_mocks'],
			'targets': targets,
			'name': str(mock_file),
		}


def task_compute_individual_wp():

	"""compute wp from individual realizations of xi_gg."""
	
	for param_file in param_files:
	
		correlation_file = txt_galaxy_autocorrelation_this_param(param_file)
		subdir = subdir_from_param_file(param_file)
		header_file = header_file_this_sim(subdir)
		
		script = "./Analysis/compute_wp.py"
		deps = [correlation_file, header_file, script]
		targets = [txt_reconstructed_wp_this_param(param_file)]

		rp_min = 0.1		# to match Sukhdeep's binning choice
		rp_max = 32.5428	# to match Sukhdeep's binning choice
		rp_nbins = 19		# to match Sukhdeep's binning choice
		pimax = 100. # Mpc/h 
		zlens = 0.3 # to get distance correction right

		yield {
			'actions': [f"python {script} {correlation_file} {header_file} {targets[0]} --rpmin {rp_min} --rpmax {rp_max} --nbins {rp_nbins} --pimax {pimax} --zlens {zlens}"],
			'file_dep': deps,
			'targets': targets,
			'name': str(correlation_file),
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
		
		if (not os.path.exists(targets[0])) or (os.path.getsize(targets[0]) == 0):
			yield {
				'actions': [f"python {script} {boxsize} {binsfile} {mock_file} {subsample} {targets[0]}"],
				'file_dep': deps,
				'task_dep': ['compute_HOD_mocks'],
				'targets': targets,
				'name': str(mock_file),
			}


def task_compute_individual_DeltaSigma():

	"""compute DS from individual realizations of xi_gm."""
	
	for param_file in param_files:
	
		correlation_file = txt_galaxy_matter_crosscorrelation_this_param(param_file)
		subdir = subdir_from_param_file(param_file)
		header_file = header_file_this_sim(subdir)
		
		script = "./Analysis/compute_DeltaSigma.py"
		deps = [correlation_file, header_file, script]
		targets = [txt_reconstructed_DeltaSigma_this_param(param_file)]
		
		rp_min = 0.1		# to match Sukhdeep's binning choice
		rp_max = 32.5428	# to match Sukhdeep's binning choice
		rp_nbins = 19		# to match Sukhdeep's binning choice

		yield {
			'actions': [f"python {script} --DS {correlation_file} {header_file} {targets[0]} --rpmin {rp_min} --rpmax {rp_max} --nbins {rp_nbins}"],
			'file_dep': deps,
			'targets': targets,
			'name': str(correlation_file),
		}	
			
### average correlation functions over stochastic realizations ###


def group_param_files():

	"""list all *.template_param files."""
	
	param_files = param_files_in_dir(param_dir)

	"""find those with is_stochastic=True set."""
	
	stochastic_param_files = {}
	for param_file in param_files:
	
		# read param_file
		myconfigparser = configparser.ConfigParser()
		myconfigparser.read(param_file)
		params = myconfigparser['params']
		
		if 'is_stochastic' in params:
		
			if params['is_stochastic'] == 'True':
			
				this_params = dict(params)
				# the seeds can differ, only the other parameters should be the same
				del this_params['seed'] 
				stochastic_param_files[param_file] = this_params

	"""group into identical sets. (this is quite ugly code, but it works.)"""
	
	param_groups = []
	param_groups_files = []
	
	for param_file, params in stochastic_param_files.items():
	
		if params in param_groups:
		
			# add to existing param_group
			idx = param_groups.index(params)
			param_groups_files[idx].append(param_file)
			
		else:
		
			param_groups.append(params)
			param_groups_files.append([param_file])

	return param_groups_files


def task_average_gg_correlation_functions():

	"""find groups of parameter files corresponding to stochastic realizations, then average
	xi_gg and xi_gm."""
	
	param_groups_files = group_param_files()

	"""average."""
	
	for param_files in param_groups_files:
	
		## compute basename from this set of files, compute output file
		
		param_files.sort()

		script = './Analysis/average_correlation_function.py'


		## yield task to average the *.xi_gg files

		xi_gg_output = txt_average_xi_gg_this_param(param_files[0])
		xi_gg_input = [str(Path(p).with_suffix('.template_param.autocorr')) for p in param_files]

		yield {
			'actions':	[f"python {script} {xi_gg_output} {' '.join(xi_gg_input)}"],
			'file_dep':	[script] + xi_gg_input,
			'task_dep':	['compute_galaxy_autocorrelation'],
			'targets':	[xi_gg_output],
			'name':		f"{xi_gg_output}",
		}

		
		## yield task to average the *.wp.txt files
		
		wp_output = txt_average_wp_this_param(param_files[0])
		wp_input = [str(Path(p).with_suffix('.template_param.wp.txt')) for p in param_files]
		
		yield {
			'actions': [f"python {script} {wp_output} {' '.join(wp_input)}"],
			'file_dep': [script] + wp_input,
			'task_dep': ['compute_individual_wp'],
			'targets': [wp_output],
			'name': f"{wp_output}",
		}
		

def task_average_gm_correlation_functions():

	"""find groups of parameter files corresponding to stochastic realizations, then average
	xi_gg and xi_gm."""
	
	param_groups_files = group_param_files()

	"""average."""
	
	for param_files in param_groups_files:
	
		## compute basename from this set of files, compute output file
		
		param_files.sort()

		script = './Analysis/average_correlation_function.py'


		## yield task to average the *.xi_gm files

		xi_gm_output = txt_average_xi_gm_this_param(param_files[0])
		xi_gm_input = [str(Path(p).with_suffix('.template_param.subsample_particles.crosscorr')) for p in param_files]
		
		yield {
			'actions':	[f"python {script} {xi_gm_output} {' '.join(xi_gm_input)}"],
			'file_dep':	[script] + xi_gm_input,
			'task_dep':	['compute_galaxy_matter_crosscorrelation'],
			'targets':	[xi_gm_output],
			'name':		f"{xi_gm_output}",
		}		


		## yield task to average the *.DeltaSigma.txt files
		
		DS_output = txt_average_DeltaSigma_this_param(param_files[0])
		DS_input = [str(Path(p).with_suffix('.template_param.DeltaSigma.txt')) for p in param_files]
		
		yield {
			'actions': [f"python {script} {DS_output} {' '.join(DS_input)}"],
			'file_dep': [script] + DS_input,
			'task_dep': ['compute_individual_DeltaSigma'],
			'targets': [DS_output],
			'name': f"{DS_output}",
		}


### compute derived quantities from the averaged galaxy correlation functions ###


def task_precompute_analytic_hod():

	"""precompute P_lin(k), dn/dM, b(M) for analytic HOD computations
		*once* for each simulation/cosmology."""

	for subdir in subdirectories:
	
		header_file = header_file_this_sim(subdir)

		pk_file = str(Path(header_file).with_suffix('.highres_linear_pk.txt'))
		massfun_file = str(Path(header_file).with_suffix('.tinker_massfun.txt'))
		halobias_file = str(Path(header_file).with_suffix('.tinker_halobias.txt'))
		
		script = './Analysis/precompute_hod.py'
		deps = [script, header_file, './Analysis/compute_hod.py']
		targets = [pk_file, massfun_file, halobias_file]
		
		yield {
			'actions': [f"python {script} {header_file} {pk_file} {massfun_file} {halobias_file}"],
			'file_dep': deps,
			'targets': targets,
			'name': f"{header_file}",
		}
	

def task_compute_wp_ratio():

	"""compute ratio: (averaged wp) / wp_analytic."""
	
	for param_file in param_files:
	
		correlation_file = txt_average_xi_gg_this_param(param_file)	
		
		if Path(correlation_file).exists():

			sim_wp_file = txt_average_wp_this_param(param_file)
			sim_xigg_file = txt_average_xi_gg_this_param(param_file)
		
			output_xigg_file = txt_analytic_xigg_this_param(param_file)
			output_xigg_ratio_file = txt_ratio_xigg_this_param(param_file)

			output_wp_file = txt_analytic_wp_this_param(param_file)
			output_wp_ratio_file = txt_ratio_wp_this_param(param_file)
			
#			sim_ds_file = 
#			output_ds_ratio_file = 
	
			subdir = subdir_from_param_file(param_file)
			header_file = header_file_this_sim(subdir)
			
			pk_file = str(Path(header_file).with_suffix('.highres_linear_pk.txt'))
			massfun_file = str(Path(header_file).with_suffix('.tinker_massfun.txt'))
			halobias_file = str(Path(header_file).with_suffix('.tinker_halobias.txt'))
			
			script = "./Analysis/compute_hod.py"
			
			deps = [sim_xigg_file, sim_wp_file, param_file, header_file, script]
			targets = [output_xigg_file, output_xigg_ratio_file,
						output_wp_file, output_wp_ratio_file]

			yield {
				'actions': [f"python {script} {header_file} {pk_file} {massfun_file} {halobias_file} {param_file} {sim_xigg_file} {output_xigg_file} {output_xigg_ratio_file} {sim_wp_file} {output_wp_file} {output_wp_ratio_file}"],
				'file_dep': deps,
				'targets': targets,
				'name': f"{sim_wp_file}",
			}

