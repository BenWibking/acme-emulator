import argparse
import numpy as np
import h5py as h5

def compute_total_size(dsname, input_filenames, output_filename):
    total_size = 0
    for input_filename in input_filenames:
        with h5.File(input_filename, mode='r') as h5in:
            total_size += h5in[dsname].size
    return total_size

def get_input_dtype(dsname, input_filenames):
    with h5.File(input_filenames[0], mode='r') as h5in:
        return h5in[dsname].dtype

def concat(dsname, input_filenames, output_filename):
    total_size = compute_total_size(dsname, input_filenames, output_filename)
    input_dtype = get_input_dtype(dsname, input_filenames)

    with h5.File(output_filename, mode='w') as h5out:
        h5out.create_dataset(dsname, (total_size,), dtype=input_dtype,
                             chunks=True, compression="gzip")
        idx = 0 # is there a more elegant solution?
        for input_filename in input_filenames:
            with h5.File(input_filename, mode='r') as h5in:
                data = h5in[dsname]
                h5out[dsname][idx:idx+data.size] = data
                idx += data.size
                h5out.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='concatenate HDF5 datasets across files into a single file.')
    parser.add_argument('dataset_name') # TODO: make this concatenate *every* dataset across the set of hdf5 files
    parser.add_argument('output_filename')
    parser.add_argument('input_filenames',nargs='*') # multiple arguments
    args = parser.parse_args()

    # output info
    print("concatenating dataset",args.dataset_name," in hdf5 files:",args.input_filenames,"...")
    print("saving output to:",args.output_filename)

    concat(args.dataset_name, args.input_filenames, args.output_filename)


