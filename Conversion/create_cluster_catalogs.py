import argparse
import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('halo_file', type = str)
parser.add_argument('ClusterDefinition', type = int)
parser.add_argument('scatter', type = float)
parser.add_argument('seed', type = int)
parser.add_argument('cluster_file', type = str)
args = parser.parse_args()

np.random.seed(args.seed)

minmass = 1e13

infile = h5py.File(args.halo_file, 'r')
clusters = infile['halos']
clusters = clusters[clusters['mass'] > minmass]
infile.close()

randnorms = np.random.normal(0.0, 1.0, int(clusters.size))

i = 0
for x in clusters:
  x['mass'] = np.exp( np.log(x['mass']) + args.scatter*randnorms[i] )
  i += 1
  
clusters.sort(order = 'mass')

clusters = clusters[-args.ClusterDefinition:]

outfile = h5py.File(args.cluster_file, 'w')
outfile.create_dataset('particles', data = clusters)
outfile.close()
