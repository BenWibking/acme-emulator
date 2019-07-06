#!/usr/bin/env python

from os import path
import argparse
import subprocess
import numpy as np
import scipy.optimize
import h5py as h5
import sys

import config
import configparser


def compute_ngal(logM, dM, mass_function, logMmin, M1_over_Mmin, M0_over_M1,
				 alpha, siglogM, f_cen):

    M = 10.**logM
    Mmin = 10.**logMmin
    M1 = M1_over_Mmin*Mmin
    M0 = M0_over_M1*M1
    mean_cen = 0.5 * (1. + scipy.special.erf((logM - logMmin)/siglogM))
    with np.errstate(invalid='ignore'):
        mean_sat = mean_cen * (((M - M0) / M1)**alpha)

    ngal = np.zeros(logM.shape)
    ngal += mean_cen * f_cen # f_cen should not affect satellites
    ngal[M > M0] += mean_sat[logM > np.log10(M0)]
    ngal *= mass_function

    return np.sum(ngal*dM)


def compute_ncen(logM, dM, mass_function, logMmin, siglogM, f_cen):

    M = 10.**logM
    Mmin = 10.**logMmin
    mean_cen = 0.5 * (1. + scipy.special.erf((logM - logMmin)/siglogM))
    ngal = np.zeros(logM.shape)
    ngal += mean_cen * f_cen
    ngal *= mass_function
    
    return np.sum(ngal*dM)


def compute_HOD_parameters(ngal=None, M1_over_Mmin=None, M0_over_M1=None, alpha=None,
						   siglogM=None, f_cen=None,
						   halos=None, header=None, mass_fun_file=None, logMmin_guess=13.2):
                           
    """
    Compute the physical (er, Msun h^-1) HOD parameters from ratios of masses
    and the desired number density of galaxies.
    We convert the number density of galaxies into Mmin using the halo mass function.
    """
    
    assert(M0_over_M1 > 0.)

    binmin, binmax, mass_function = np.loadtxt(mass_fun_file, unpack=True)
    
    assert(~np.any( np.isnan(mass_function) ))
    assert(~np.any( np.isnan(binmin) ))
    assert(~np.any( np.isnan(binmax) ))
    
    dM = binmax - binmin
    M = 0.5*(binmax+binmin)
    logM = np.log10(M)

    """
    now solve for logMmin:
    compute_ngal(...) - ndens = 0
    """
    
    this_ngal = lambda logMmin: compute_ngal(logM, dM, mass_function, logMmin,
    										 M1_over_Mmin, M0_over_M1, alpha, siglogM, f_cen)
    objective = lambda logMmin: this_ngal(logMmin) - ngal
    
    logMmin = scipy.optimize.newton(objective, logMmin_guess, maxiter = 100)
    Mmin = 10**(logMmin)
    M1 = Mmin*M1_over_Mmin
    M0 = M1*M0_over_M1
    
    print('logMmin: %s' % logMmin, file=sys.stderr)
    print('optimized ngal: %s' % this_ngal(logMmin), file=sys.stderr)
    print('desired ngal: %s' % ngal, file=sys.stderr)

    return logMmin, np.log10(M0), np.log10(M1)
    

def compute_HOD_parameters_centralsonly(ncen=None, siglogM=None, f_cen=None,
                                        halos=None, header=None, mass_fun_file=None,
                                        logMmin_guess=13.5):
                                        
    binmin, binmax, mass_function = np.loadtxt(mass_fun_file, unpack=True)
    dM = binmax - binmin
    M = 0.5*(binmax+binmin)
    logM = np.log10(M)

    this_ncen = lambda logMmin: compute_ncen(logM, dM, mass_function, logMmin, siglogM, f_cen)
    objective = lambda logMmin: this_ncen(logMmin) - ncen
    
    logMmin = scipy.optimize.newton(objective, logMmin_guess, maxiter = 100)
    Mmin = 10**(logMmin)
    
    print('logMmin: %s' % logMmin, file=sys.stderr)
    print('optimized ncen: %s' % this_ncen(logMmin), file=sys.stderr)
    print('desired ncen: %s' % ncen, file=sys.stderr)
    
    return logMmin


def populate_hod(halo_file, galaxy_file, env_file,
                 omega_m,redshift,boxsize,siglogM,logMmin,logM0,logM1,alpha,
                 q_env,del_gamma,f_cen,A_conc,delta_b,delta_c,R_rescale,is_stochastic,seed):
                 
    script_path = path.dirname(path.abspath(__file__))+"/../cHOD/compute_mocks"
    
    cmd_line = [script_path,str(omega_m),str(redshift),
                str(siglogM),str(logMmin),str(logM0),str(logM1),
                str(alpha),str(q_env),str(del_gamma),str(f_cen),
                str(A_conc),str(delta_b),str(delta_c),str(R_rescale),
                str(boxsize),
                halo_file,galaxy_file,env_file,str(is_stochastic),str(seed)]
                
    print(' '.join(cmd_line),file=sys.stderr)
    
    subprocess.call(cmd_line)
    

def compute(halo_file,env_file,header_file,output_file,
            siglogM=None,logMmin=None,logM0=None,logM1=None,alpha=None,
            q_env=None,del_gamma=None,f_cen=None,
            A_conc=None,delta_b=None,delta_c=None,R_rescale=None,
            is_stochastic=None,seed=None):
            
    cf = config.AbacusConfigFile(header_file)
    boxsize = cf.boxSize
    omeganow_m = cf.OmegaNow_m
    omega_m = cf.Omega_M
    redshift = cf.redshift

    ## now compute HOD
    
    populate_hod(halo_file, output_file, env_file,
                 omega_m,redshift,boxsize,siglogM,logMmin,logM0,logM1,alpha,
                 q_env,del_gamma,f_cen,A_conc,delta_b,delta_c,R_rescale,
                 int(is_stochastic),seed)

    ## read back output file
    
    with h5.File(output_file, mode='r') as mock:
        vol = boxsize**3
        ngal_obtained = np.sum(mock['particles']['weight']) / vol
        print('obtained ngal: %s' % ngal_obtained, file=sys.stderr)



if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	
	parser.add_argument('hod_params_path')
	parser.add_argument('header_path')
	parser.add_argument('halo_path')
	parser.add_argument('output_path')
	parser.add_argument('env_path')
	parser.add_argument('mass_fun_path')
	parser.add_argument('--centrals_only',default=False,action='store_true')
	args = parser.parse_args()
	
	
	## read meta-HOD parameters
	
	myconfigparser = configparser.ConfigParser()
	myconfigparser.read(args.hod_params_path)
	params = myconfigparser['params']
	
	f_cen = float(params['f_cen'])
	A_conc = float(params['A_conc'])
	delta_b = float(params['delta_b'])
	delta_c = float(params['delta_c'])
	
	if 'R_rescale' in params.keys():
		R_rescale = float(params['R_rescale'])
	else:
		R_rescale = 1.0
		
	stochastic = False
	seed = 42
	
	if 'is_stochastic' in params:
	    stochastic = bool(params['is_stochastic'])
	    if 'seed' in params:
	        seed = int(params['seed'])
	
	
	## find HOD parameters
	
	if args.centrals_only == False:
	
	    logMmin, logM0, logM1 = compute_HOD_parameters(ngal=float(params['ngal']),
	                                                   siglogM=float(params['siglogm']),
	                                                   M0_over_M1=float(params['m0_over_m1']),
	                                                   M1_over_Mmin=float(params['m1_over_mmin']),
	                                                   alpha=float(params['alpha']),
	                                                   f_cen=f_cen,
	                                                   halos=args.halo_path,
	                                                   header=args.header_path,
	                                                   mass_fun_file=args.mass_fun_path)
	
	    compute(args.halo_path,args.env_path,args.header_path,args.output_path,
	            params['siglogm'],logMmin,logM0,logM1,params['alpha'],params['q_env'],
	            params['del_gamma'],f_cen,A_conc,delta_b,delta_c,R_rescale,
	            stochastic,seed=seed)
	            
	else:
	
	    # only compute HOD for centrals, *assuming the number density of centrals is fixed*
	    logMmin = compute_HOD_parameters_centralsonly(ncen=float(params['ncen']),
	                                                  siglogM=float(params['siglogm']),
	                                                  f_cen=f_cen,
	                                                  halos=args.halo_path,
	                                                  header=args.header_path,
	                                                  mass_fun_file=args.mass_fun_path)
	                                                  
	    logM0 = 14. # these sat. parameters will be ignored
	    logM1 = 15.
	    
	    compute(args.halo_path,args.env_path,args.header_path,args.output_path,
	            params['siglogm'],logMmin,logM0,logM1,params['alpha'],
	            params['q_env'],params['del_gamma'],f_cen,A_conc,delta_b,delta_c,R_rescale,
	            stochastic,seed=seed)
	    
	


