import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob

import argparse
import configparser

myconfigparser = configparser.ConfigParser()

def readfile(filename):
    x = []
    y = []
    infile = open(filename, 'r')
    infile.readline()
    for line in infile:
        line = line.strip()
        columns = line.split()
        x.append((float(columns[0]) + float(columns[1]))/2)
        y.append(float(columns[3]))
    infile.close()
    return x, y


def is_excluded_dir(x):
    return (str(x)[0]=='_' or str(x)[0]=='.')

def recursive_iter(p):
    yield p
    for subdir in p.iterdir():
        if subdir.is_dir() and not is_excluded_dir(subdir):
            yield from recursive_iter(subdir)

parser = argparse.ArgumentParser()

parser.add_argument('redshift', type = str)
parser.add_argument('directory', type = str)

args = parser.parse_args()

dirs =  list(recursive_iter(Path(args.directory)))
redshift = args.redshift

plt.figure(figsize = (10,10))

for subdir in dirs: #Plotting Matter-Matter Cosmology Comparison
    if str(subdir).split('/')[-1] == redshift:
        x = readfile(str(subdir)+"/particles_subsample.autocorr")[0]
        y = readfile(str(subdir)+"/particles_subsample.autocorr")[1]
        plt.plot(x, y, marker = 'o', linestyle = '-', label = "Cosmology "+str(str(subdir).split('/')[-2]))
        plt.xlim(min(x), max(x))
plt.xlabel("$r [h^{-1}Mpc]$", fontsize = '28')
plt.ylabel("$\\xi_{MM}(r)$", fontsize = '28')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-4,1e5)
plt.legend(loc = 0, fontsize = '20')
plt.savefig(args.directory+"/Xi_MM_"+redshift+".pdf")
plt.gcf().clear()

plt.figure(figsize = (10,10))

for subdir in dirs: #Plotting Cluster-Matter Cosmology Comparison 
    if str(subdir).split('/')[-1] == redshift:
        x = readfile(str(subdir)+"/2.0e+14subsample_particles.crosscorr")[0]
        y = readfile(str(subdir)+"/2.0e+14subsample_particles.crosscorr")[1]
        plt.plot(x, y, marker = 'o', linestyle = '-', label = "Cosmology "+str(str(subdir).split('/')[-2]))
        plt.xlim(min(x), max(x))
plt.xlabel("$r [h^{-1}Mpc]$", fontsize = '28')
plt.ylabel("$\\xi_{CM}(r)$", fontsize = '28')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-4,1e5)
plt.legend(loc = 0, fontsize = '20')
plt.savefig(args.directory+"/Xi_CM_"+redshift+".pdf")
plt.gcf().clear()

plt.figure(figsize = (10,10))

for subdir in dirs: #Plotting Galaxy-Galaxy Cosmology Comparison
    if str(subdir).split('/')[-1] == redshift:
        resultfiles = []
        for file in glob.glob(str(subdir)+'/NHOD_*.param.autocorr'):
            myconfigparser.read(file)
            if myconfigparser['params']['label'] == "Fiducial":
                resultfiles.append(str(file))
        y = np.zeros(len(x)) 
        for mock in resultfiles:
            y += np.array(readfile(mock)[1])/len(resultfiles)
            x = readfile(mock)[0]
        plt.plot(x, y, marker = 'o', linestyle = '-', label = "Cosmology "+str(str(subdir).split('/')[-2]))
        plt.xlim(min(x), max(x))
plt.xlabel("$r [h^{-1}Mpc]$", fontsize = '28')
plt.ylabel("$\\xi_{GG}(r)$", fontsize = '28')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-4,1e5)
plt.legend(loc = 0, fontsize = '20')
plt.savefig(args.directory+"/Xi_GG_"+redshift+".pdf")
plt.gcf().clear()

plt.figure(figsize = (10,10))

for subdir in dirs: #Plotting Galaxy-Matter Cosmology Comparison
    if str(subdir).split('/')[-1] == redshift:
        resultfiles = []
        for file in glob.glob(str(subdir)+'/NHOD_*subsample_particles.crosscorr'):
            myconfigparser.read(file)
            if myconfigparser['params']['label'] == "Fiducial":
                resultfiles.append(str(file))
        y = np.zeros(len(x)) 
        for mock in resultfiles:
            y += np.array(readfile(mock)[1])/len(resultfiles)
            x = readfile(mock)[0]
        plt.plot(x, y, marker = 'o', linestyle = '-', label = "Cosmology "+str(str(subdir).split('/')[-2]))
        plt.xlim(min(x), max(x))
plt.xlabel("$r [h^{-1}Mpc]$", fontsize = '28')
plt.ylabel("$\\xi_{GM}(r)$", fontsize = '28')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-4,1e5)
plt.legend(loc = 0, fontsize = '20')
plt.savefig(args.directory+"/Xi_GM_"+redshift+".pdf")
plt.gcf().clear()

plt.figure(figsize = (10,10))

for subdir in dirs: #Plotting Cluster-Galaxy Cosmology Comparison
    if str(subdir).split('/')[-1] == redshift:
        resultfiles = []
        for file in glob.glob(str(subdir)+'/NHOD_*clusters.crosscorr'):
            myconfigparser.read(file)
            if myconfigparser['params']['label'] == "Fiducial":
                resultfiles.append(str(file))
        y = np.zeros(len(x)) 
        for mock in resultfiles:
            y += np.array(readfile(mock)[1])/len(resultfiles)
            x = readfile(mock)[0]
        plt.plot(x, y, marker = 'o', linestyle = '-', label = "Cosmology "+str(str(subdir).split('/')[-2]))
        plt.xlim(min(x), max(x))
plt.xlabel("$r [h^{-1}Mpc]$", fontsize = '28')
plt.ylabel("$\\xi_{CG}(r)$", fontsize = '28')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-4,1e5)
plt.legend(loc = 0, fontsize = '20')
plt.savefig(args.directory+"/Xi_CG_"+redshift+".pdf")
plt.gcf().clear()

plt.figure(figsize = (10,10))

for subdir in dirs: #Plotting Galaxy-Galaxy R Cosmology Comparison
    if str(subdir).split('/')[-1] == redshift:
        MM = readfile(str(subdir)+"/particles_subsample.autocorr")[1]
        bins = readfile(str(subdir)+"/particles_subsample.autocorr")[0]
        resultfiles = []
        for file in glob.glob(str(subdir)+'/NHOD_*.param.autocorr'):
            myconfigparser.read(file)
            if myconfigparser['params']['label'] == "Fiducial":
                resultfiles.append(str(file))
        y = np.zeros(len(bins)) 
        for mock in resultfiles:
            y += np.array(readfile(mock)[1])/len(resultfiles)
        R = np.sqrt(y/np.array(MM))
        plt.plot(bins, R, marker = 'o', linestyle = '-', label = "Cosmology "+str(str(subdir).split('/')[-2]))
        plt.xlim(min(x), max(x))
plt.xlabel("$r [h^{-1}Mpc]$", fontsize = '28')
plt.ylabel("$R_{GG}(r)$", fontsize = '28')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1,30)
plt.legend(loc = 0, fontsize = '20')
plt.savefig(args.directory+"/R_GG_"+redshift+".pdf")
plt.gcf().clear()

plt.figure(figsize = (10,10))

for subdir in dirs: #Plotting Galaxy-Matter R Cosmology Comparison
    if str(subdir).split('/')[-1] == redshift:
        MM = readfile(str(subdir)+"/particles_subsample.autocorr")[1]
        bins = readfile(str(subdir)+"/particles_subsample.autocorr")[0]
        resultfiles = []
        for file in glob.glob(str(subdir)+'/NHOD_*subsample_particles.crosscorr'):
            myconfigparser.read(file)
            if myconfigparser['params']['label'] == "Fiducial":
                resultfiles.append(str(file))
        y = np.zeros(len(bins)) 
        for mock in resultfiles:
            y += np.array(readfile(mock)[1])/len(resultfiles)
        R = np.sqrt(y/np.array(MM))
        plt.plot(bins, R, marker = 'o', linestyle = '-', label = "Cosmology "+str(str(subdir).split('/')[-2]))
        plt.xlim(min(x), max(x))
plt.xlabel("$r [h^{-1}Mpc]$", fontsize = '28')
plt.ylabel("$R_{GM}(r)$", fontsize = '28')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1,30)
plt.legend(loc = 0, fontsize = '20')
plt.savefig(args.directory+"/R_GM_"+redshift+".pdf")
plt.gcf().clear()

plt.figure(figsize = (10,10))

for subdir in dirs: #Plotting Cluster-Galaxy R Cosmology Comparison
    if str(subdir).split('/')[-1] == redshift:
        MM = readfile(str(subdir)+"/particles_subsample.autocorr")[1]
        bins = readfile(str(subdir)+"/particles_subsample.autocorr")[0]
        resultfiles = []
        for file in glob.glob(str(subdir)+'/NHOD_*clusters.crosscorr'):
            myconfigparser.read(file)
            if myconfigparser['params']['label'] == "Fiducial":
                resultfiles.append(str(file))
        y = np.zeros(len(bins)) 
        for mock in resultfiles:
            y += np.array(readfile(mock)[1])/len(resultfiles)
        R = np.sqrt(y/np.array(MM))
        plt.plot(bins, R, marker = 'o', linestyle = '-', label = "Cosmology "+str(str(subdir).split('/')[-2]))
        plt.xlim(min(x), max(x))
plt.xlabel("$r [h^{-1}Mpc]$", fontsize = '28')
plt.ylabel("$R_{CG}(r)$", fontsize = '28')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1,30)
plt.legend(loc = 0, fontsize = '20')
plt.savefig(args.directory+"/R_CG_"+redshift+".pdf")
plt.gcf().clear()

plt.figure(figsize = (10,10))

for subdir in dirs: #Plotting Cluster-Matter R Cosmology Comparison
    if str(subdir).split('/')[-1] == redshift:
        MM = readfile(str(subdir)+"/particles_subsample.autocorr")[1]
        bins = readfile(str(subdir)+"/2.0e+14subsample_particles.crosscorr")[0]
        CM = readfile(str(subdir)+"/2.0e+14subsample_particles.crosscorr")[1]
        R = np.sqrt(np.array(CM)/np.array(MM))
        plt.plot(bins, R, marker = 'o', linestyle = '-', label = "Cosmology "+str(str(subdir).split('/')[-2]))
        plt.xlim(min(x), max(x))
plt.xlabel("$r [h^{-1}Mpc]$", fontsize = '28')
plt.ylabel("$R_{CM}(r)$", fontsize = '28')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1,30)
plt.legend(loc = 0, fontsize = '20')
plt.savefig(args.directory+"/R_CM_"+redshift+".pdf")
plt.gcf().clear()
