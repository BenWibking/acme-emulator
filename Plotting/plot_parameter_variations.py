import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('output_file',help='plot pdf')
parser.add_argument('title', help='plot title')
parser.add_argument('-f','--input_file',nargs=2,action='append',help='results files for a given parameter variations')

args = parser.parse_args()

f = args.input_file
outfile = args.output_file
title = args.title

def load_correlation_file(filename):
	table = np.loadtxt(filename,unpack=False)
	binmin, binmax, counts, R = [table[:,i] for i in range(4)]   
	bins = [x/2 for x in np.add(binmin, binmax)]
	return bins, R

bins, R1 = load_correlation_file(f[0][0]) #for our purposes this is almost always the fiducial cosmology/HOD
bins, R2 = load_correlation_file(f[1][0]) 
bins, R3 = load_correlation_file(f[2][0])

label1 = f[0][1]
label2 = f[1][1]
label3 = f[2][1]

#import sys
#print(f[0][1], file=sys.stderr)
#print(f[1][1], file=sys.stderr)
#print(f[2][1], file=sys.stderr)

linewidth = 2
plt.plot(bins, R1/R1, color = 'red', marker = 'o', linewidth = linewidth, label = label1)
plt.plot(bins, R2/R1, color = 'green', marker = 'o', linewidth = linewidth, label = label2)
plt.plot(bins, R3/R1, color = 'blue', marker = 'o', linewidth = linewidth, label = label3)

fontsize = 20
plt.ylabel(r"$R = \sqrt{\frac{\xi_{gg}}{\xi_{mm}}}$", fontsize = fontsize)
plt.xlabel(r"$r$ $\mathrm{[h^{-1} Mpc]}$", fontsize = fontsize)
plt.title(title, fontsize = fontsize)
	
plt.xlim(min(bins), max(bins))

rvalue_min = [min(R) for R in [R1, R2, R3]]
rvalue_max = [max(R) for R in [R1, R2, R3]]
#plt.ylim(min(rvalue_min), max(rvalue_max))

plt.xscale('log')
#plt.yscale('log')
plt.grid('on')

plt.legend()

plt.savefig(outfile)
