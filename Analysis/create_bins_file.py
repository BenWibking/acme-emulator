import numpy as np
import argparse

if __name__ == '__main__':
        parser = argparse.ArgumentParser()

        parser.add_argument('rmin', type = float)
        parser.add_argument('rmax', type = float)
        parser.add_argument('nbins', type = int)
        parser.add_argument('output_file', type = str)

        args = parser.parse_args()

        nbins = args.nbins
        rmin = args.rmin
        rmax = args.rmax

        edges = np.logspace(np.log10(rmin), np.log10(rmax), num = nbins + 1)

        outfile = open(args.output_file, 'w')

        i = 0
        while i < len(edges)-1:
	        outfile.write(str(edges[i])+"  "+str(edges[i+1])+"\n")
	        i += 1

        outfile.close()
