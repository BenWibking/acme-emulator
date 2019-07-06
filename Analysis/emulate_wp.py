import numpy as np
import argparse
import configparser
from emulator import load_derivative_file, emulate, emulator_data_from_files
from compute_wp import wp

"""steps:
1. emulate b_g
2. emulate b_nl
3. compute pk via pycamb
4. compute xi_mm_linear via FAST-PT
5. compute xi_gg = b_g**2 * (b_nl**2 * xi_mm_linear)
6. compute wp
"""

use_log_parameter = ['ngal','ncen','siglogM','M1_over_Mmin','alpha','H0','sigma_8','M0_over_M1','Omega_M']#,'f_cen']

def read_param_config_file(param_file):
        """read param config file *.template_param"""
        myconfigparser = configparser.ConfigParser()
        myconfigparser.read(param_file)
        params = myconfigparser['params']
        return params

def read_emulator_data(deriv_files, fiducial_model_file, fiducial_param_config, input_param_config):
        # read derivatives w.r.t. each parameter and fiducial parameter values
        derivs = []
        fiducial_param_values = np.zeros(len(deriv_files))
        input_param_values = np.zeros(len(deriv_files))
        for i, (deriv_file, param_name) in enumerate(deriv_files):
                binmin,binmax,derivative = load_derivative_file(deriv_file)
                derivs.append(derivative)
                if param_name in use_log_parameter:
                        fiducial_param_values[i] = np.log(float(fiducial_param_config[param_name]))
                        input_param_values[i] = np.log(float(input_param_config[param_name]))
                else:
                        fiducial_param_values[i] = float(fiducial_param_config[param_name])
                        input_param_values[i] = float(input_param_config[param_name])

        # read fiducial model values
        binmin, binmax, null, fiducial_model_values = np.loadtxt(fiducial_model_file, unpack=True)

        return binmin, binmax, derivs, fiducial_param_values, input_param_values, fiducial_model_values

def emulate_bias_with_files(deriv_files, fiducial_file, fiducial_param_config, input_param_config):
        binmin, binmax, deriv, fiducial_param_values, input_param_values, fiducial = read_emulator_data(deriv_files,
                                                                                        fiducial_file,
                                                                                        fiducial_param_config,
                                                                                        input_param_config)
        bias_model = emulate(fiducial, deriv, input_param_values, fiducial_param_values)
        return binmin, binmax, bias_model

def emulate_wp_with_files(fiducial_bg_file, deriv_bg_files, fiducial_bnl_file, deriv_bnl_files,
                          fiducial_param_file, input_param_file, output_wp_file, xi_mm_linear_file):

        ## (0) read input parameter values (input_param_values = ...)
        fiducial_param_config = read_param_config_file(fiducial_param_file)
        input_param_config = read_param_config_file(input_param_file)
        emulated_Omega_m = float(input_param_config['Omega_m'])
        if 'z_lens' in input_param_config:
                emulated_z_lens = float(input_param_config['z_lens'])
        else:
                emulated_z_lens = 0.27 # default


        ## (1) emulate ln b_g, convert to b_g
        binmin_bg, binmax_bg, ln_bg_model = emulate_bias_with_files(deriv_bg_files,
                                                                 fiducial_bg_file,
                                                                 fiducial_param_config,
                                                                 input_param_config)
        bg_model = np.exp(ln_bg_model)


        ## (2) emulate ln b_nl, convert to b_nl
        binmin_bnl, binmax_bnl, ln_bnl_model = emulate_bias_with_files(deriv_bnl_files,
                                                                    fiducial_bnl_file,
                                                                    fiducial_param_config,
                                                                    input_param_config)
        bnl_model = np.exp(ln_bnl_model)

        # sanity check
        tol=1.0e-5
        assert (((binmin_bg-binmin_bnl)<tol).all()) and (((binmax_bg-binmax_bnl)<tol).all())


        ##   (3) compute pk for cosmology given in input_param_config
        ##   (4) compute xi_mm_linear via FAST-PT
        ## [ONLY IF INPUT COSMOLOGY DOES NOT MATCH FIDUCIAL COSMOLOGY]
        if not (fiducial_param_config['Omega_M'] == input_param_config['Omega_M'] and fiducial_param_config['sigma_8'] == input_param_config['sigma_8']):
                assert False # not yet implemented
        ## [ELSE]
        else:
        ##   (3+4) read xi_mm_linear from given file
                assert xi_mm_linear_file != None
                binmin_mm, binmax_mm, xi_mm_lin = load_derivative_file(xi_mm_linear_file)
                assert (((binmin_bg-binmin_mm)<tol).all()) and (((binmax_bg-binmax_mm)<tol).all())


        ## (5) compute xi_gg
        xi_gg = bg_model**2 * ( bnl_model**2 * xi_mm_lin )
        # [save xi_gg to file]
        #np.savetxt(output_xi_gg_file, np.c_[binmin_mm, binmax_mm, np.zeros(xi_gg.shape[0]), xi_gg], delimiter='\t')

        ## (5) compute wp
        binmin_wp, binmax_wp, this_wp = wp(binmin_mm, binmax_mm, xi_gg, emulated_Omega_m, emulated_z_lens)


        ## (7) save model output
        np.savetxt(output_wp_file, np.c_[binmin_wp, binmax_wp, np.zeros(this_wp.shape[0]), this_wp], delimiter='\t')


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
#        parser.add_argument('emulator_settings_file')
        parser.add_argument('fiducial_param_file',help='*.template_param file describing fiducial parameters')
        parser.add_argument('fiducial_bg_file',help='fiducial b_g file')
        parser.add_argument('--deriv_bg_file',help='derivative file, param name for b_g',nargs=2,action='append',required=True)
        parser.add_argument('fiducial_bnl_file',help='fiducial b_nl file')
        parser.add_argument('--deriv_bnl_file',help='derivative file, param name for b_nl',nargs=2,action='append',required=True)
        parser.add_argument('--xi_mm_linear',default=None,help='tabulated linear matter autocorrelation file')

        parser.add_argument('input_param_file',help='*.template_param file describing emulated parameters')
        parser.add_argument('output_wp_file',help='txt file output for emulated wp')

        args = parser.parse_args()
        
        emulate_wp_with_files(args.fiducial_bg_file, args.deriv_bg_file,
                              args.fiducial_bnl_file, args.deriv_bnl_file,
                              args.fiducial_param_file,
                              args.input_param_file,
                              args.output_wp_file,
                              args.xi_mm_linear)
