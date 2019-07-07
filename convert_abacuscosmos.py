from doit.tools import run_once
from pipeline_defs import *
from pathlib import Path
import os

rockstar_subdirectories = list(recursive_iter(working_directory / 'Rockstar'))
FOF_subdirectories = list(recursive_iter(working_directory / 'FOF'))

print(working_directory)
print(list(recursive_iter(working_directory / 'Rockstar')))

def task_convert_binary_particles_to_hdf5():

	"""convert binary particle files to hdf5 table"""

	for subdir in FOF_subdirectories:

		inputs = [str(Path(subdir) / x) for x in ['field_subsamples.bin', 'halo_subsamples.bin']]
		combined_bin = binary_particles_this_sim(subdir)

		# find header file in ../../Rockstar/xx/z0.xxx/header
		header = header_file_this_sim(subdir).replace("FOF", "Rockstar")

		script = "./Conversion/convert_abacuscosmos_particles_to_hdf5.py"

		deps = [header, script]
		targets = [hdf5_particles_this_sim(subdir)]

		if not os.path.exists(targets[0]):
				yield {
						'actions': [f"cat {' '.join(inputs)} > {combined_bin}",
									f"python {script} {combined_bin} {header} {targets[0]}",
									f"rm {' '.join(inputs)} {combined_bin}"],
						'file_dep': deps,
						'targets': targets,
						'uptodate': [run_once], # don't re-run since we deleted inputs
						'name': subdir
				}


def task_subsample_hdf5_particles():

	"""subsample particles in hdf5 table; output to hdf5 table"""

	for subdir in FOF_subdirectories:

		particles = hdf5_particles_this_sim(subdir)
		subsample = hdf5_particles_subsample_this_sim(subdir)
		script = "./Conversion/subsample_particles.py"

		deps = [particles, script]
		targets = [subsample]
		
		yield {
				'actions': [f"python {script} {particles} {subsample}"],
				'file_dep': deps,
				'targets': targets,
				'name': subdir,
		}


def task_subset_hdf5_Rockstar_halos():

	"""use only host halos, copy only needed properties to reduce file size."""

	for subdir in rockstar_subdirectories:

		fullcat = hdf5_Rockstar_allprops_catalog_this_sim(subdir)
		subcat = hdf5_Rockstar_catalog_this_sim(subdir)
		script = "./Conversion/convert_Rockstar_hdf5_to_hdf5.py"

		deps = [fullcat, script]
		targets = [subcat]
	
		yield {
				'actions': [f"python {script} {fullcat} {subcat}"],
				'file_dep': deps,
				'targets': targets,
				'name': subdir,
		}
