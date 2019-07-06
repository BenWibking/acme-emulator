DOIT_CONFIG = {
    'minversion': '0.24.0', # minimum version of doit needed to run this file
    'backend': 'sqlite3',
    'dep_file': 'doit-db.sqlite3', #saves return values of actions in this file
}

import sys
if sys.version_info >= (3, 6):
    pass
else:
    sys.stdout.write("Sorry, requires Python 3.6+\n")
    sys.exit(1)

import configparser
from pathlib import Path

## read in parameters

from doit import get_var
sample = get_var('sample',None)
print("sample = {}".format(sample))

if sample == 'sdss21':
    boxsize = 720.0
    working_directory = Path('./simulations/Snapshots/720Mpc_Cosmo0')
    redshift = 'z0.100'
    rmax = 110.0
    rmin = 0.01
    nbins = 80
    param_dir = str(Path('./Params/SDSS_Main'))

elif sample == 'cmass-emulator':
    boxsize = 720.0
    working_directory = Path('./AbacusCosmos_720box')
    redshift = 'z0.500'
    rmax = 110.0
    rmin = 0.01
    nbins = 80
    param_dir = str(Path('./Params/CMASS_emulator'))

elif sample == 'lowz-01':
    boxsize = 720.0
    working_directory = Path('./AbacusCosmos_720box')
    redshift = 'z0.100'
    rmax = 110.0
    rmin = 0.01
    nbins = 80
    param_dir = str(Path('./Params/LOWZ_emulator_01'))

elif sample == 'lowz-03':
    boxsize = 720.0
    working_directory = Path('./AbacusCosmos_720box')
    redshift = 'z0.300'
    rmax = 110.0
    rmin = 0.01
    nbins = 80
    param_dir = str(Path('./Params/LOWZ_emulator_03'))

elif sample == 'lowz-05':
    boxsize = 720.0
    working_directory = Path('./AbacusCosmos_720box')
    redshift = 'z0.500'
    rmax = 110.0
    rmin = 0.01
    nbins = 80
    param_dir = str(Path('./Params/LOWZ_emulator_05'))
    
elif sample == 'lowz-phases-03':
    boxsize = 720.0
    working_directory = Path('./AbacusCosmos_phases')
    redshift = 'z0.300'
    rmax = 110.0
    rmin = 0.01
    nbins = 80
    param_dir = str(Path('./Params/LOWZ_phases_03'))
    
elif sample == 'abacuscosmos-z07':
    boxsize = 720.0
    working_directory = Path('./AbacusCosmos_720box')
    redshift = 'z0.700'
    rmax = 110.0
    rmin = 0.01
    nbins = 80
    param_dir = str(Path('./Params/CMASS_emulator_07'))
    
elif sample == 'cmass-emulator-phases':
    boxsize = 720.0
    working_directory = Path('./AbacusCosmos_phases')
    redshift = 'z0.500'
    rmax = 110.0
    rmin = 0.01
    nbins = 80
    param_dir = str(Path('./Params/CMASS_emulator_phases'))

elif sample == 'lowz':
    boxsize = 720.0
    working_directory = Path('./simulations/Snapshots/720Mpc_Cosmo0')
    redshift = 'z0.300'
    rmax = 110.0
    rmin = 0.01
    nbins = 80
    param_dir = str(Path('./Params/LOWZ'))

#elif sample == 'lowz-emulator':
#    boxsize = 720.0
#    working_directory = Path('./simulations/Snapshots/emulator_720')
#    redshift = 'z0.300'
#    rmax = 110.0
#    rmin = 0.01
#    nbins = 80
#    param_dir = str(Path('./Params/LOWZ_emulator'))

elif sample == 'lowz-hod-test':
    boxsize = 720.0
    working_directory = Path('./simulations/Snapshots/720Mpc_Cosmo0')
    redshift = 'z0.300'
    rmax = 110.0
    rmin = 0.01
    nbins = 80
    param_dir = str(Path('./Params/LOWZ_HOD_tests'))

elif sample == 'lowz-hod-test-singleparameter':
    boxsize = 720.0
    working_directory = Path('./simulations/Snapshots/720Mpc_Cosmo0')
    redshift = 'z0.300'
    rmax = 110.0
    rmin = 0.01
    nbins = 80
    param_dir = str(Path('./Params/LOWZ_HOD_tests_singleparameter'))

elif sample == 'lowz-hod-test-1sigma':
    boxsize = 720.0
    working_directory = Path('./simulations/Snapshots/720Mpc_Cosmo0')
    redshift = 'z0.300'
    rmax = 110.0
    rmin = 0.01
    nbins = 80
    param_dir = str(Path('./Params/LOWZ_HOD_tests_sampleposterior_1sigma'))

elif sample == 'lowz-hod-test-2sigma':
    boxsize = 720.0
    working_directory = Path('./simulations/Snapshots/720Mpc_Cosmo0')
    redshift = 'z0.300'
    rmax = 110.0
    rmin = 0.01
    nbins = 80
    param_dir = str(Path('./Params/LOWZ_HOD_tests_sampleposterior_2sigma'))

elif sample == 'lowz-hod-test-3sigma':
    boxsize = 720.0
    working_directory = Path('./simulations/Snapshots/720Mpc_Cosmo0')
    redshift = 'z0.300'
    rmax = 110.0
    rmin = 0.01
    nbins = 80
    param_dir = str(Path('./Params/LOWZ_HOD_tests_sampleposterior_3sigma'))

elif sample == 'lowz-concentration-test':
    boxsize = 720.0
    working_directory = Path('./simulations/Snapshots/720Mpc_Cosmo0')
    redshift = 'z0.300'
    rmax = 110.0
    rmin = 0.01
    nbins = 80
    param_dir = str(Path('./Params/LOWZ_concentration_test'))

elif sample == 'redmagic':
    boxsize = 720.0
    working_directory = Path('./simulations/Snapshots/720Mpc_Cosmo0')
    redshift = 'z0.300'
    rmax = 110.0
    rmin = 0.01
    nbins = 80
    param_dir = str(Path('./Params/redmagic'))

elif sample == 'redmagic-phases':
    boxsize = 720.0
    working_directory = Path('./phases_plummer')
    redshift = 'z0.300'
    rmax = 110.0
    rmin = 0.01
    nbins = 80
    param_dir = str(Path('./Params/redmagic-phases'))

else:
    print("Must specify a boxsize defined in pipeline_defs.py")
    exit(1)

halos = get_var('halos',None)
if halos == 'FOF':
#    exit("You specified FOF halos. This is a mistake.")
    halo_working_directory = working_directory / 'FOF'
else:
    halo_working_directory = working_directory / 'Rockstar'

## convenience functions

def param_files_in_dir(subdir):
    return [str(x) for x in Path(subdir).glob('NHOD_*') if x.suffix=='.template_param']

def param_files_in_dir_centralsonly(subdir):
    return [str(x) for x in Path(subdir).glob('NHOD_*') if x.suffix=='.template_param_centralsonly']

def subdir_from_param_file(param_file):
    myconfigparser = configparser.ConfigParser()
    myconfigparser.read(param_file)
    params = myconfigparser['params']
    subdir = str(params['dir']).strip('"')
    return working_directory / 'Rockstar' / subdir / redshift


def is_redshift_dir(x):
    return (x.name.startswith(redshift))

def is_excluded_dir(x):
    return (str(x)[0]=='_' or str(x)[0]=='.' or x.name=='Rockstar' or x.name=='FOF' or x.name=='profiling')

def recursive_iter(p):
    if not is_excluded_dir(p) and is_redshift_dir(p):
        yield p
    for subdir in p.iterdir():
        if subdir.is_dir() and not is_excluded_dir(subdir):
            yield from recursive_iter(subdir)


subdirectories = list(recursive_iter(halo_working_directory)) # list() is crucial; iterators can't be re-used!

def template_param_input_list(template_dir):
    return [str(x) for x in Path(template_dir).glob('NHOD_*') if x.suffix=='.template_param']


#******************************
#***** Particles and halo catalogs *****
#******************************

def binfile_this_sim(rmin, rmax, subdir):
    return str(subdir)+'/bins'+str(rmin)+'to'+str(rmax)+'.txt'

def mass_function_this_sim(subdir):
    return str(subdir)+'/RShalos_mass_function.txt'

def hdf5_Rockstar_catalog_this_file(halo_file):
    return str(halo_file)+'.hdf5'

def hdf5_Rockstar_catalog_this_sim(subdir):
    return str(subdir)+'/RShalos.hdf5'

def hdf5_Rockstar_allprops_catalog_this_sim(subdir):
    return str(subdir)+'/RShalos_allprops.hdf5'

def hdf5_Rockstar_env_this_sim(subdir):
    return str(subdir)+'/RShalos_env.hdf5'

def binary_particles_this_sim(subdir):
    return str(subdir)+'/particles.bin'

def header_file_this_sim(subdir):
    return str(subdir)+'/header'

def hdf5_particles_this_sim(subdir):
    return str(subdir)+'/particles.hdf5'

def hdf5_particles_subsample_this_sim(subdir):
    return str(subdir)+'/particles_subsample.hdf5'

def hdf5_particles_fixed_this_sim(subdir):
    return str(subdir)+'/particles.hdf5.fixed'

def hdf5_particles_subsample_fixed_this_sim(subdir):
    return str(subdir)+'/particles_subsample.hdf5.fixed'

def hdf5_FOF_catalog_this_file(halo_file):
    return str(halo_file)+'.hdf5' # should manipulate pathlib obj instead?

def hdf5_FOF_catalog_this_sim(subdir):
    return str(subdir)+'/halos.hdf5'

def hdf5_Rockstar_catalog_this_file(halo_file):
    return str(halo_file)+'.hdf5'

#*******************#*******************#*******************
#***** Results that do NOT depend on galaxy population *****
#*******************#*******************#*******************

def txt_linear_matter_power_this_sim(subdir):
    return str(subdir)+'/../camb_matterpower.dat'

def txt_matter_autocorrelation_this_sim(subdir):
    return str(subdir)+'/particles_subsample.autocorr'

def txt_linear_matter_autocorrelation_this_sim(subdir):
    return str(subdir)+'/linear_theory_matter.autocorr'    

def txt_nonlinear_matter_bias_this_sim(subdir):
    return str(subdir)+'/nonlinear_matter_bias.txt'    

def txt_ln_bnl_this_sim(subdir):
    return str(subdir)+'/ln_bnl.txt'

#*******************#*******************#*******************
#***** Results (that *do* depend on galaxy population) *****
#*******************#*******************#*******************

def hdf5_HOD_mock_this_param(param_file):
    return str(param_file)+'.mock.hdf5'

def hdf5_HOD_logMmin_this_param(param_file):
    return str(param_file)+'.logMmin.txt'

def hdf5_HOD_mock_weighted_this_param(param_file):
    return str(param_file)+'.weighted_mock.hdf5'

def hdf5_HOD_mock_this_param_centralsonly(param_file):
    return str(param_file)+'.centralsonly.mock.hdf5'
    
def txt_analytic_xigg_this_param(param_file):
	return str(param_file)+'.analytic_xigg.txt'
	
def txt_analytic_wp_this_param(param_file):
	return str(param_file)+'.analytic_wp.txt'
	
def txt_ratio_wp_this_param(param_file):
	return str(param_file)+'.ratio_wp.txt'

def txt_galaxy_number_density_this_param(mock_file):
    return str(mock_file)+'.ngal'

def txt_galaxy_number_density_this_param_centralsonly(mock_file):
    return str(mock_file)+'.centralsonly.ngal'

def txt_log_galaxy_number_density_this_param(mock_file):
    return str(mock_file)+'.ln_ngal'

def txt_log_galaxy_number_density_this_param_centralsonly(mock_file):
    return str(mock_file)+'.centralsonly.ln_ngal'

def txt_galaxy_autocorrelation_this_param(mock_file):
    return str(mock_file)+'.autocorr'

def txt_galaxy_autocorrelation_weighted_this_param(mock_file):
    return str(mock_file)+'.weighted_autocorr'

def txt_galaxy_autocorrelation_this_param_centralsonly(mock_file):
    return str(mock_file)+'.centralsonly.autocorr'

def txt_galaxy_matter_crosscorrelation_this_param(param_file):
    return str(param_file)+'.subsample_particles.crosscorr'

def txt_galaxy_matter_crosscorrelation_weighted_this_param(param_file):
    return str(param_file)+'.subsample_particles.weighted_crosscorr'

def txt_galaxy_matter_crosscorrelation_this_param_centralsonly(param_file):
    return str(param_file)+'.centralsonly.subsample_particles.crosscorr'

def txt_galaxy_bias_this_param(param_file):
    return str(param_file)+'.galaxy_bias.txt'

def txt_galaxy_bias_weighted_this_param(param_file):
    return str(param_file)+'.weighted_galaxy_bias.txt'

def txt_galaxy_bias_this_param_centralsonly(param_file):
    return str(param_file)+'.centralsonly.galaxy_bias.txt'

def txt_ln_bg_this_param(param_file):
    return str(param_file)+'.ln_bg.txt'

def txt_ln_bg_this_param_centralsonly(param_file):
    return str(param_file)+'.centralsonly.ln_bg.txt'

def txt_galaxy_matter_correlation_coefficient_this_param(param_file):
    return str(param_file)+'.galaxy_matter_corr_coef.txt'

def txt_galaxy_matter_correlation_coefficient_weighted_this_param(param_file):
    return str(param_file)+'.weighted_galaxy_matter_corr_coef.txt'

def txt_galaxy_matter_correlation_coefficient_this_param_centralsonly(param_file):
    return str(param_file)+'.centralsonly.galaxy_matter_corr_coef.txt'

def txt_ln_rgm_this_param(param_file):
    return str(param_file)+'.ln_rgm.txt'

def txt_ln_rgm_this_param_centralsonly(param_file):
    return str(param_file)+'.centralsonly.ln_rgm.txt'

def txt_average_xi_gg_this_param(param_file):
    return str(param_file)+'.average_xi_gg.txt'

def txt_average_xi_gm_this_param(param_file):
    return str(param_file)+'.average_xi_gm.txt'

def txt_reconstructed_xi_gg_this_param(param_file):
    return str(param_file)+'.xi_gg.txt'

def txt_reconstructed_xi_gg_this_param_centralsonly(param_file):
    return str(param_file)+'.centralsonly.xi_gg.txt'

def txt_reconstructed_xi_gm_this_param(param_file):
    return str(param_file)+'.xi_gm.txt'

def txt_reconstructed_xi_gm_this_param_centralsonly(param_file):
    return str(param_file)+'.centralsonly.xi_gm.txt'

def txt_reconstructed_wp_this_param(param_file):
    return str(param_file)+'.wp.txt'
    
def txt_average_wp_this_param(param_file):
	return str(param_file)+'.average_wp.txt'
	
def txt_average_DeltaSigma_this_param(param_file):
	return str(param_file)+'.average_DeltaSigma.txt'

def txt_ratio_xigg_this_param(param_file):
	return str(param_file)+'.ratio_xigg.txt'

def txt_smoothed_bias_this_param(param_file):
    return str(param_file)+'.smoothed_bias.txt'

def txt_smoothed_bias_weighted_this_param(param_file):
    return str(param_file)+'.weighted_smoothed_bias.txt'

def txt_wp_weighted_this_param(param_file):
    return str(param_file)+'.weighted_wp.txt'

def txt_reconstructed_wp_this_param_centralsonly(param_file):
    return str(param_file)+'.centralsonly.wp.txt'

def txt_reconstructed_DeltaSigma_this_param(param_file):
    return str(param_file)+'.DeltaSigma.txt'

def txt_DeltaSigma_weighted_this_param(param_file):
    return str(param_file)+'.weighted_DeltaSigma.txt'

def txt_reconstructed_DeltaSigma_baldauf_this_param(param_file):
    return str(param_file)+'.DeltaSigma_baldauf.txt'

def txt_reconstructed_DeltaSigma_baldauf_this_param_centralsonly(param_file):
    return str(param_file)+'.centralsonly.DeltaSigma_baldauf.txt'

def txt_reconstructed_DeltaSigma_this_param_centralsonly(param_file):
    return str(param_file)+'.centralsonly.DeltaSigma.txt'

def txt_reconstructed_log_wp_this_param(param_file):
    return str(param_file)+'.ln_wp.txt'

def txt_reconstructed_log_wp_this_param_centralsonly(param_file):
    return str(param_file)+'.centralsonly.ln_wp.txt'

def txt_reconstructed_log_DeltaSigma_this_param(param_file):
    return str(param_file)+'.ln_DeltaSigma.txt'

def txt_reconstructed_log_DeltaSigma_baldauf_this_param(param_file):
    return str(param_file)+'.ln_DeltaSigma_baldauf.txt'

def txt_reconstructed_log_DeltaSigma_baldauf_this_param_centralsonly(param_file):
    return str(param_file)+'.centralsonly.ln_DeltaSigma_baldauf.txt'

def txt_reconstructed_log_DeltaSigma_this_param_centralsonly(param_file):
    return str(param_file)+'.centralsonly.ln_DeltaSigma.txt'

def txt_emulated_wp_this_param(param_file):
    return str(param_file)+'.emulated_wp.txt'

def txt_fractional_accuracy_emulated_wp_this_param(param_file):
    return str(param_file)+'.accuracy_emulated_wp.txt'

def txt_emulated_DS_this_param(param_file):
    return str(param_file)+'.emulated_DS.txt'

def txt_fractional_accuracy_emulated_DS_this_param(param_file):
    return str(param_file)+'.accuracy_emulated_DS.txt'

#********************
#***** Plotting *****
#********************

def pdf_matter_autocorrelation_this_sim(subdir):
    return str(subdir)+'/particles_subsample.autocorr.pdf'

def pdf_galaxy_autocorrelation_this_param(param_file):
    return str(param_file)+'.autocorr.pdf'

def pdf_galaxy_crosscorrelation_this_param(param_file):
    return str(param_file)+'.crosscorr.pdf'

def pdf_galaxy_bias_this_param(param_file):
    return str(param_file)+'.galaxy_bias.pdf'

def pdf_galaxy_matter_correlation_coefficient_this_param(param_file):
    return str(param_file)+'.galaxy_matter_corr_coef.pdf'

def pdf_linear_matter_autocorrelation_this_sim(subdir):
    return str(subdir)+'/linear_theory_matter.autocorr.pdf'    

def pdf_nonlinear_matter_bias_this_sim(subdir):
    return str(subdir)+'/nonlinear_matter_bias.pdf'    

def pdf_xi_plot_this_sim(subdir):
    return str(subdir)+'/Xi.pdf'

def pdf_R_plot_this_sim(subdir):
    return str(subdir)+'/R.pdf'
    
def pdf_R_plot_this_redshift(directory, redshift, name):
    return str(directory)+'/R_'+str(name)+'_z'+str(redshift)+'.pdf'
    
def pdf_Xi_plot_this_redshift(directory, redshift, name):
    return str(directory)+'/Xi_'+str(name)+'_z'+str(redshift)+'.pdf'
    
def pdf_cosmo_comparison_this_redshift(directory, redshift):
    return str(directory)+'/cosmo_comp_z'+str(redshift)+'.pdf'

def pdf_Rgg_parameter_variation(param):
    return './Derivatives/Rgg_vary_'+param+'.pdf'

def pdf_Rgm_parameter_variation(param):
    return './Derivatives/Rgm_vary_'+param+'.pdf'

def plot_label_this_param(param_file):
    myconfigparser = configparser.ConfigParser()
    myconfigparser.read(str(param_file))
    params = myconfigparser['params']
    string = params['label']
    if string.startswith('"') and string.endswith('"'):
        string = string[1:-1]
    string = '$'+string+'$'
    return string

def parameter_value_this_param(parameter, param_file):
    myconfigparser = configparser.ConfigParser()
    ret = []

    try:
        myconfigparser.read(str(param_file))
        params = myconfigparser['params']

        if parameter == 'combined_om_s8':
            sigma_8 = float(params['sigma_8'])
            Omega_M = float(params['Omega_M'])
            value = sigma_8 * Omega_M**0.36
            ret = str(value)
        else:
            ret = params[parameter]

    except Exception:
        print("Couldn't read file ", str(param_file))
        print("while attempting to read parameter: ", parameter)
        raise

    return ret


