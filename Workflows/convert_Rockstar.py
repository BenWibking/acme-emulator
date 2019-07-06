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

def task_convert_Rockstar_halos_to_hdf5():
    """convert binary Rockstar halo catalog to hdf5 table"""

    def input_list(subdir):
        return [str(x) for x in Path(subdir).glob('halos_*') if x.suffix=='.bin']

    for subdir in subdirectories:
        halo_files = input_list(subdir)
        header = header_file_this_sim(subdir)
        script = "./Conversion/convert_Rockstar_binary_to_hdf5.py"

        for halo_file in halo_files:
            deps = [halo_file, script]
            targets = [hdf5_Rockstar_catalog_this_file(halo_file)]

            yield {
                'actions': ["python %(script)s %(halo_file)s %(target)s"
                            % {"script": script, "halo_file": halo_file, "target": targets[0]}],
                'file_dep': deps,
                'targets': targets,
                'name': str(halo_file),
                }

def task_concat_hdf5_Rockstar_halos():
    """concatenate hdf5 Rockstar halo files.
    N.B. This will *not* run on the first invokation of this script because input_list will return nothing.  This needs to be fixed."""
    from itertools import chain

    def input_list(subdir):
        return [str(x) for x in Path(subdir).glob('halos_*') if x.suffix=='.hdf5']

    for subdir in subdirectories:
        halo_files = input_list(subdir)
        script = "./Conversion/concat_hdf5_datasets_across_files.py"
        hdf5_dataset = "halos"

        deps = list(chain(*[halo_files, [script]]))
        targets = [hdf5_Rockstar_catalog_this_sim(subdir)]

        if halo_files != []:
            yield {
                'actions': ["python %(script)s %(dataset)s %(target)s %(halo_files)s"
                            % {"script": script, "halo_files": ' '.join(halo_files), "dataset": hdf5_dataset, "target": targets[0]}],
                'file_dep': deps,
                'task_dep': ['convert_Rockstar_halos_to_hdf5'],
                'targets': targets,
                'name': subdir,
                }

