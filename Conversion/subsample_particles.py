import argparse
import numpy as np
import h5py

class subsampleHDF5:
    def __init__(self,dsname,filename,inputFile,subsampleFrac):
        self.dsname = dsname
        self.filename = filename
        self.input = inputFile

        self.length = self.get_data_len(self.input)
        
        if(subsampleFrac > 1):
            return Exception("subsample fraction must be less than 1.")
        if(subsampleFrac <= 0):
            return Exception("subsample fraction must be positive.")

        self.subsampleLength = int(self.length * subsampleFrac)
        print("data length: "+str(self.length))
        print("subsample length: "+str(self.subsampleLength))

    def get_data_len(self,filename):
        # open filename as HDF5 file
        with h5py.File(filename,'r') as f:
            return f[self.dsname].len()

    def run(self):
        # based on https://gist.github.com/zonca/8e0dda9d246297616de9
        with h5py.File(self.filename, mode='w') as h5f:
            print("Processing {0}".format(self.input))
            f = h5py.File(self.input,'r')
            data = f[self.dsname]
            print("Data read, length {0}".format(data.len()))

            if(data.len() != self.length):
                return Exception("data length does not match!")

            subsampleIdx = np.random.choice(self.length, size=self.subsampleLength, replace=False)
            bool_array = np.zeros(self.length,dtype=bool)
            bool_array[subsampleIdx] = True
            subsampleData = data[bool_array]

            try:
                h5f[self.dsname][:] = subsampleData
            except KeyError: #if dataset not created yet
                h5f.create_dataset(self.dsname, data=subsampleData, maxshape=(None,))
                h5f[self.dsname].resize((self.subsampleLength,))
 
            h5f.flush()
            f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='subsample hdf5 particle file.')
    parser.add_argument('particle_filename')
    parser.add_argument('output_filename')
    args = parser.parse_args()

    # output info
    print("subsampling hdf5 file:",args.particle_filename,"...")
    print("saving output to:",args.output_filename)

    # construct a 1% subsample of the input file (which itself is a 10% subsample of the total)
    subsample = subsampleHDF5('particles',args.output_filename,args.particle_filename,0.01)
    subsample.run()

