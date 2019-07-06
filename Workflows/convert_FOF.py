from pipeline_defs import *
from pathlib import Path

DOIT_CONFIG = {
    'minversion': '0.24.0', # minimum version of doit needed to run this file
    'backend': 'json',
    'dep_file': 'doit-db.json', # saves return values of 'actions' in this file
}

"""
This script is run with `doit`, a python automation tool similar to make.
It depends on a particular directory structure, to bo documented here.

(remember, task_ functions generate tasks. Tasks are separate functions
[*or* shell commands, in which case this does not apply] that do NOT begin with task_)
"""

def task_concat_binary_files():
    """concatenate binary particle files"""

    def input_list(subdir):
        return [str(x) for x in Path(subdir).glob('*.field')]+[str(x) for x in Path(subdir).glob('*.particles')]

    for subdir in subdirectories:
        deps = input_list(subdir)
        targets = [binary_particles_this_sim(subdir)]
        if deps != []:
            yield {
                'actions': ["cat %(dependencies)s > %(target)s" % {"dependencies": ' '.join(deps), "target": targets[0]}],  
                'file_dep': deps,
                'targets': targets,
                'name': subdir
                }

def task_convert_binary_particles_to_hdf5():
    """convert binary particle file to hdf5 table"""

    for subdir in subdirectories:
        particles = binary_particles_this_sim(subdir)
        header = header_file_this_sim(subdir)
        script = "./Conversion/convert_particles_to_hdf5.py"

        deps = [particles, header, script]
        targets = [hdf5_particles_this_sim(subdir)]

        if Path(particles).exists():
            yield {
                'actions': ["python %(script)s %(particles)s %(header)s %(target)s"
                        % {"script": script, "particles": particles, "header": header, "target": targets[0]}],
                'file_dep': deps,
                'targets': targets,
                'name': subdir,
                }

def task_subsample_hdf5_particles():
    """subsample particles in hdf5 table; output to hdf5 table"""

    for subdir in subdirectories:
        particles = hdf5_particles_this_sim(subdir)
        subsample = hdf5_particles_subsample_this_sim(subdir)
        script = "./Conversion/subsample_particles.py"

        deps = [particles, script]
        targets = [subsample]
        
        if Path(particles).exists():
            yield {
                'actions': ["python %(script)s %(particles)s %(subsample)s"
                        % {"script": script, "particles": particles, "subsample": subsample}],
                'file_dep': deps,
                'targets': targets,
                'name': subdir,
                }

def task_convert_FOF_halos_to_hdf5():
    """convert binary halo catalog to hdf5 table"""

    def input_list(subdir):
        return [str(x) for x in Path(subdir).glob('halos_*') if x.suffix=='']

    for subdir in subdirectories:
        halo_files = input_list(subdir)
        header = header_file_this_sim(subdir)
        script = "./Conversion/convert_FOF_halos_to_hdf5.py"

        for halo_file in halo_files:
            deps = [halo_file, header, script]
            targets = [hdf5_FOF_catalog_this_file(halo_file)]

            yield {
                'actions': ["python %(script)s %(halo_file)s %(header)s %(target)s"
                            % {"script": script, "halo_file": halo_file, "header": header, "target": targets[0]}],
                'file_dep': deps,
                'targets': targets,
                'name': str(subdir)+'/'+str(halo_file),
                }

def task_concat_hdf5_FOF_halos():
    """concatenate hdf5 FOF halo files."""
    from itertools import chain

    def input_list(subdir):
        return [str(x) for x in Path(subdir).glob('halos_*') if x.suffix=='.hdf5']

    for subdir in subdirectories:
        halo_files = input_list(subdir)
        script = "./Conversion/concat_hdf5_datasets_across_files.py"
        hdf5_dataset = "halos"

        deps = list(chain(*[halo_files, [script]]))
        targets = [hdf5_FOF_catalog_this_sim(subdir)]

        print(deps)

        if halo_files != []:
            yield {
                'actions': ["python %(script)s %(dataset)s %(target)s %(halo_files)s"
                            % {"script": script, "halo_files": ' '.join(halo_files), "dataset": hdf5_dataset, "target": targets[0]}],
                'file_dep': deps,
                'targets': targets,
                'name': subdir,
                }
