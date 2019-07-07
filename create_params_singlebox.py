import numpy as np
import argparse
import configparser
from pathlib import Path
from create_params_LOWZ_emu import write_param_file


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('fiducial_param_file')
	parser.add_argument('simulation_dir')
	parser.add_argument('output_filename_base')
	parser.add_argument('--nseeds', default=20)
	args = parser.parse_args()
	
	
	## read fiducial param file
	
	fiducial_config = configparser.ConfigParser()
	fiducial_config.read(args.fiducial_param_file)

	param_names = list(fiducial_config['params'].keys())
	print(f"param_names: {param_names}")
	fiducial_params = []

	for i, param_name in enumerate(param_names):
		if param_name in ['dir', 'is_stochastic', 'seed']:
			continue
		fiducial_params.append(fiducial_config['params'][param_name])

	new_params = zip(param_names, fiducial_params)
	
	sim_dir = Path(args.simulation_dir)
	nparams = len(param_names)

	
	## write out param files with new seeds
	
	write_param_file(args.output_filename_base,
					 fiducial_config,
					 list(new_params),
					 sim_dir.name,
					 index=0, nseeds=int(args.nseeds),
					 overwrite_files=False)
	
