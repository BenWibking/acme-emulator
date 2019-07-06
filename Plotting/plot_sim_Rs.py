import matplotlib.pyplot as plt
import numpy as np
import os.path as path
import glob

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('matter_matter', type = str)
parser.add_argument('cluster_matter', type = str)
parser.add_argument('subdir', type = str)
parser.add_argument('fig1_name', type = str)
parser.add_argument('fig2_name', type = str)

args = parser.parse_args()

subdir = args.subdir

def readfile(filename):
    x = []
    y = []
    infile = open(path.abspath(filename), 'r')
    infile.readline()
    for line in infile:
        line = line.strip()
        columns = line.split()
        x.append((float(columns[0]) + float(columns[1]))/2)
        y.append(float(columns[3]))
    infile.close()
    return np.array(x), np.array(y)
    
cluster_galaxies = []
for file in glob.glob(str(subdir)+'/NHOD_*clusters.crosscorr'):
    cluster_galaxies.append(str(file))

galaxy_galaxies = []
for file in glob.glob(str(subdir)+'/NHOD_*.param.autocorr'):
    galaxy_galaxies.append(str(file))

galaxy_matters = []
for file in glob.glob(str(subdir)+'/NHOD_*subsample_particles.crosscorr'):
    galaxy_matters.append(str(file))

bins = readfile(args.matter_matter)[0]
XiMM = readfile(args.matter_matter)[1]

XiCM = readfile(args.cluster_matter)[1]

XiGG = np.zeros(len(bins))
XiCG = np.zeros(len(bins))
XiGM = np.zeros(len(bins))

for mock in cluster_galaxies:
    XiCG += readfile(path.abspath(mock))[1] / len(cluster_galaxies)
 
for mock in galaxy_galaxies:
    XiGG += readfile(path.abspath(mock))[1] / len(galaxy_galaxies)

for mock in galaxy_matters:
    XiGM += readfile(path.abspath(mock))[1] / len(galaxy_matters)

Rcg = np.sqrt(XiCG/np.array(XiMM))
Rcm = np.sqrt(XiCM/np.array(XiMM))
Rgg = np.sqrt(XiGG/np.array(XiMM))
Rgm = np.sqrt(XiGM/np.array(XiMM))

plt.figure(figsize = (10,10))

plt.plot(bins, XiCG, color = 'red', marker = 'o', linestyle = '-', label = "$\mathrm{Cluster-Galaxy}$")
plt.plot(bins, XiCM, color = 'green', marker = 'o', linestyle = '-', label = "$\mathrm{Cluster-Matter}$")
plt.plot(bins, XiGG, color = 'blue', marker = 'o', linestyle = '-', label = "$\mathrm{Galaxy-Galaxy}$")
plt.plot(bins, XiGM, color = 'yellow', marker = 'o', linestyle = '-', label = "$\mathrm{Galaxy-Matter}$")
plt.plot(bins, XiMM, color = 'purple', marker = 'o', linestyle = '-', label = "$\mathrm{Matter-Matter}$")
plt.xlim(min(bins), max(bins))
plt.ylim(1e-4,1e5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("$r [h^{-1}Mpc]$", fontsize = '28')
plt.ylabel("$\\xi(r)$", fontsize = '28')
plt.title(str(subdir), fontsize = '28')
plt.legend(loc = 0, fontsize = '20')

plt.savefig(args.fig1_name)

plt.figure(figsize = (10,10))

plt.plot(bins, Rcg, color = 'red', marker = 'o', linestyle = '-', label = "$\mathrm{Cluster-Galaxy}$")
plt.plot(bins, Rcm, color = 'green', marker = 'o', linestyle = '-', label = "$\mathrm{Cluster-Matter}$")
plt.plot(bins, Rgg, color = 'blue', marker = 'o', linestyle = '-', label = "$\mathrm{Galaxy-Galaxy}$")
plt.plot(bins, Rgm, color = 'yellow', marker = 'o', linestyle = '-', label = "$\mathrm{Galaxy-Matter}$")
plt.xlim(min(bins), max(bins))
plt.ylim(1e0,3e1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("$r [h^{-1}Mpc]$", fontsize = '28')
plt.ylabel("$R(r) = \sqrt{\\frac{\\xi(r)}{\\xi_{mm}(r)}}$", fontsize = '28')
plt.title(str(subdir), fontsize = '28')
plt.legend(loc = 0, fontsize = '20')

plt.savefig(args.fig2_name)
