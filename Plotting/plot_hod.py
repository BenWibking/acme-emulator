#!/usr/bin/env python
import argparse
import numpy as np
import scipy.optimize
import h5py as h5
import sys
import matplotlib.pyplot as plt

import configparser
from plot_cumulative_parameter_constraints import pretty_print_label

def mean_sat_conditional(M, M0, M1, alpha):
    sat = np.zeros(M.shape)
    sat[M > M0] = (((M[M > M0] - M0) / M1)**alpha)
    return sat

def compute_ngal(logM, dM, mass_function, logMmin, M1_over_Mmin, M0_over_M1, alpha, siglogM):
    M = 10.**logM
    Mmin = 10.**logMmin
    M1 = M1_over_Mmin*Mmin
    M0 = M0_over_M1*M1
    mean_cen = 0.5 * (1. + scipy.special.erf((logM - logMmin)/siglogM))
    mean_sat = mean_cen * mean_sat_conditional(M, M0, M1, alpha)

    ngal = np.zeros(logM.shape)
    ngal += mean_cen
    ngal[logM > np.log10(M0)] += mean_sat[logM > np.log10(M0)]
    ngal *= mass_function

    return np.sum(ngal*dM)

def compute_HOD_parameters(ngal=None, M1_over_Mmin=None, M0_over_M1=None, alpha=None, siglogM=None,
                           mass_fun_file=None, logMmin_guess=13.5):
    """
    Compute the physical (er, Msun h^-1) HOD parameters from ratios of masses
    and the desired number density of galaxies.
    We convert the number density of galaxies into Mmin using the halo mass function.
    """
    binmin, binmax, mass_function = np.loadtxt(mass_fun_file, unpack=True)
    dM = binmax - binmin
    M = 0.5*(binmax+binmin)
    logM = np.log10(M)

    """
    now solve for logMmin:
    compute_ngal(...) - ndens = 0
    """
    this_ngal = lambda logMmin: compute_ngal(logM, dM, mass_function, logMmin, M1_over_Mmin, M0_over_M1, alpha, siglogM)
    objective = lambda logMmin: this_ngal(logMmin) - ngal
    
    logMmin = scipy.optimize.newton(objective, logMmin_guess, maxiter = 100)
    Mmin = 10**(logMmin)
    M1 = Mmin*M1_over_Mmin
    M0 = M1*M0_over_M1
    
    print('logMmin: %s' % logMmin, file=sys.stderr)
    print('optimized ngal: %s' % this_ngal(logMmin), file=sys.stderr)
    print('desired ngal: %s' % ngal, file=sys.stderr)

    return logMmin, np.log10(M0), np.log10(M1)

def plot_this_hod(siglogM,logMmin,logM0,logM1,alpha,plot_mmin,plot_mmax,plot_split,label):
    logM = np.linspace(plot_mmin,plot_mmax,100)

    mean_cen = 0.5 * (1. + scipy.special.erf((logM - logMmin)/siglogM))

    M = 10.**logM
    Mmin = 10.**logMmin
    M1 = 10.**logM1
    M0 = 10.**logM0
    mean_sat = mean_cen * mean_sat_conditional(M, M0, M1, alpha)

    hod = mean_cen + mean_sat
    if plot_split:
        plt.plot(M, mean_cen, '--', label='centrals', color='C0')
        plt.plot(M, mean_sat, '-.', label='satellites', color='C1')
        plt.plot(M, hod, color='black')
    else:
        plt.plot(M, hod, label=label)

def plot_hod(hod_params_files,mass_fun_file,output_file,plot_mmin=12.,plot_mmax=16.,plot_split=True):
    plt.figure()

    for hod_params_file in hod_params_files:
        # read meta-HOD parameters
        myconfigparser = configparser.ConfigParser()
        myconfigparser.read(hod_params_file)
        params = myconfigparser['params']
        ngal = float(params['ngal'])
        siglogM = float(params['siglogm'])
        M0_over_M1 = float(params['m0_over_m1'])
        M1_over_Mmin = float(params['m1_over_mmin'])
        alpha = float(params['alpha'])
        q_env = float(params['q_env'])
        del_gamma = float(params['del_gamma'])
        parameter = params['parameter']
        if parameter != 'None':
            label = "%s = %s" % (pretty_print_label(parameter), params[parameter])
        else:
            label = None

        # find HOD parameters
        logMmin, logM0, logM1 = compute_HOD_parameters(ngal=float(params['ngal']),siglogM=float(params['siglogm']),M0_over_M1=float(params['m0_over_m1']),M1_over_Mmin=float(params['m1_over_mmin']),alpha=float(params['alpha']),mass_fun_file=args.mass_fun_path)

        if parameter == 'q_env':
            # special case: q_env
            # adjust to show HOD for density extremes
            logMmin_highdens = logMmin + q_env * (1 - 0.5)
            logMmin_lowdens = logMmin + q_env * (0 - 0.5)
            logM1_highdens = np.log10(M1_over_Mmin) + logMmin_highdens
            logM1_lowdens = np.log10(M1_over_Mmin) + logMmin_lowdens
            logM0_highdens = np.log10(M0_over_M1) + logM1_highdens
            logM0_lowdens = np.log10(M0_over_M1) + logM1_lowdens

            plot_this_hod(siglogM,logMmin_highdens,logM0_highdens,logM1_highdens,alpha,
                          plot_mmin,plot_mmax,plot_split,label + " (high density environment)")
            plot_this_hod(siglogM,logMmin_lowdens,logM0_lowdens,logM1_lowdens,alpha,
                          plot_mmin,plot_mmax,plot_split,label + " (low density environment)")
        else:
            plot_this_hod(siglogM,logMmin,logM0,logM1,alpha,
                          plot_mmin,plot_mmax,plot_split,label)

    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(10.**plot_mmin, 10.**plot_mmax)
    plt.ylim(0.1, 100.)
    plt.xlabel(r"halo mass $M_h$ ($h^{-1} M_{\odot}$)")
    plt.ylabel(r"mean number of galaxies $\langle N | M_h \rangle$")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(output_file)


parser = argparse.ArgumentParser()
parser.add_argument('--plot_split',default=False,action='store_true')
parser.add_argument('mass_fun_path')
parser.add_argument('output_path')
parser.add_argument('hod_params_path',nargs='*')

args = parser.parse_args()

plot_hod(args.hod_params_path, args.mass_fun_path, args.output_path, plot_split=args.plot_split)

