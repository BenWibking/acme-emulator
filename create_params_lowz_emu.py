import numpy as np
import configparser
from pathlib import Path
import hashlib
import re

def tex_escape(text):

	"""
		:param text: a plain text message
		:return: the message escaped to appear correctly in LaTeX
	"""

	conv = {
		'&': r'\&',
		'%': r'\%',
		'$': r'\$',
		'#': r'\#',
		'_': r'\_',
		'{': r'\{',
		'}': r'\}',
		'~': r'\textasciitilde{}',
		'^': r'\^{}',
		'\\': r'\textbackslash{}',
		'<': r'\textless',
		'>': r'\textgreater',
	}

	regex = re.compile('|'.join(re.escape(str(key)) for key in sorted(conv.keys(), key = lambda item: - len(item))))
	return regex.sub(lambda match: conv[match.group()], text)


def write_param_file(base_filename, fiducial_config, new_params, file_id, nseeds=5, index=None,
					overwrite_files=False):

	"""write out parameter file with parameters 'params'.
	[1. read in fiducial parameter file]
	2. change varied parameters
	3. write out new file
	"""

	for i in range(nseeds):

		seed = 42+i

		fiducial_config['params']['dir'] = str(file_id)
		for param,value in new_params:
			print(f"\t{param} {value}")
			fiducial_config['params'][param] = str(value)

		fiducial_config['params']['is_stochastic'] = 'True'
		fiducial_config['params']['seed'] = str(seed)

		output_filename = f"{base_filename}.{file_id}.{index}.seed_{seed}.template_param"

		if not Path(output_filename).exists() or overwrite_files==True:
			print(f"writing to {output_filename}")
			with open(output_filename,'w') as output_file:
				fiducial_config.write(output_file)

		else:
			print("parameter file already exists! exiting.")
			exit(1)


def main(args):

	"""write out parameter files for random samples."""

	nsamples = int(args.number_per_sim)


	## read fiducial param file

	fiducial_config = configparser.ConfigParser()
	fiducial_config.read(args.fiducial_param_file)

	sample_config = configparser.ConfigParser()
	sample_config.read(args.sampling_param_file)

	param_names = sample_config['settings']['sample_params'].strip().split(' ')
	
	if param_names == ['']:
		param_names = []

	fiducial_params = np.zeros(len(param_names))
	param_min = np.zeros(len(param_names))
	param_max = np.zeros(len(param_names))
	
	for i, param_name in enumerate(param_names):
		fiducial_params[i] = fiducial_config['params'][param_name]
		param_min[i] = sample_config['params_min'][param_name]
		param_max[i] = sample_config['params_max'][param_name]

	sim_dirs = [p for p in Path(args.simulation_base_dir).iterdir() if p.is_dir()]
	nsims = len(sim_dirs)
	nparams = len(param_names)


	## compute latin hypercube or uniform random samples

	if args.sample=='lhs':

		import diversipy.hycusampling as lhs
		N_total_samples = nsamples * nsims

		## this 'improved' LH sampling tends to have worse emulator performance...
		#		lhc = lhs.improved_lhd_matrix(N_total_samples, nparams)
		#		sample_array = lhs.edge_lhs(lhc) # deterministic mapping

		lhc = lhs.lhd_matrix(N_total_samples, nparams)
		sample_array = lhs.transform_perturbed(lhc) # random mapping

		new_params_values_arr = param_min + sample_array*(param_max - param_min)

		for j, sim_dir in enumerate(sim_dirs):

			for i in range(nsamples):

				idx = j*nsamples + i
				delta_param = sample_array[idx,:]
				new_param_values = param_min + delta_param*(param_max - param_min)
				new_params = zip(param_names, new_param_values)
				write_param_file(args.output_filename_base,
								fiducial_config, list(new_params), sim_dir.name,
								index=i, nseeds=args.nseeds,
								overwrite_files=args.overwrite_files)

	elif args.sample=='uniform_random':

		for j, sim_dir in enumerate(sim_dirs):

			try:
				myseed = int(sim_dir.name)
			except ValueError:
				myseed = int(hashlib.md5(str(sim_dir.name).encode()).hexdigest(),base=16) % 2**32
			
			print("simulation number: {}".format(myseed))
			RandomGenerator = np.random.RandomState(seed=myseed) # make reproducible randomness

			for i in range(nsamples):

				# uniform random sample individual parameters from [param_min, param_max]
				
				delta_param = RandomGenerator.rand(len(param_names))
				new_param_values = param_min + delta_param*(param_max - param_min)
				new_params = zip(param_names, new_param_values)
				
				write_param_file(args.output_filename_base,
								fiducial_config, new_params, sim_dir.name,
								index=i, nseeds=args.nseeds,
								overwrite_files=args.overwrite_files)
								
	elif args.sample=='none':
	
		## just replicate the same parameter file across the set of simulations
		
		assert(nsamples == 1)  # if we are just copying a single set of params, nsamples==1
		
		for j, sim_dir in enumerate(sim_dirs):

			new_param_values = fiducial_params
			new_params = zip(param_names, new_param_values)
			
			write_param_file(args.output_filename_base,
							fiducial_config, list(new_params), sim_dir.name,
							index=0, nseeds=args.nseeds,
							overwrite_files=args.overwrite_files)
		
	else:

		print("sampling strategy not specified!")
		exit(1)


if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser()
        
	parser.add_argument('--fiducial_param_file',
                            default='./Params/NHOD_lowz_fiducial_params.template_param')
	parser.add_argument('--sampling_param_file',
                            default='./Params/lowz_param_ranges_varyRvir-wide-allcosmo.txt')
	parser.add_argument('--output_filename_base',
                            default='./Params/LOWZ_HOD/NHOD_lowz')
	parser.add_argument('--simulation_base_dir',
                            default='./AbacusCosmos/AbacusCosmos_720box/Rockstar')
	
	parser.add_argument('--number_per_sim', type=int, default=10)
	parser.add_argument('--nseeds', type=int, default=20)
	parser.add_argument('--sample', default='lhs', help='sampling strategy')
	parser.add_argument('--overwrite_files', default=False, action='store_true')

	args = parser.parse_args()

	main(args)
