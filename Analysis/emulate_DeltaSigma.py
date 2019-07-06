import numpy as np
import argparse
import configparser
from emulator import load_derivative_file, emulate, emulator_data_from_files
from emulate_wp import read_param_config_file, read_emulator_data, emulate_bias_with_files, use_log_parameter
from compute_DeltaSigma import compute_gamma_t

"""steps:
1. emulate b_g
2. emulate b_nl
3. emulate r_gm
4. compute pk via pycamb [NOT IMPLEMENTED]
5. compute xi_mm_linear via FAST-PT [implemented in compute_linear_matter_correlation.py]
7. compute xi_gm = b_g * r_gm * (b_nl**2 * xi_mm_linear)
8. compute DS
"""

def emulate_DS_with_files(fiducial_bg_file, deriv_bg_files, fiducial_bnl_file, deriv_bnl_files,
                          fiducial_rgm_file, deriv_rgm_files,
                          fiducial_param_file, input_param_file, output_DS_file, xi_mm_linear_file):

        ## (0) read input parameter values (input_param_values = ...)
        fiducial_param_config = read_param_config_file(fiducial_param_file)
        input_param_config = read_param_config_file(input_param_file)
        emulated_Omega_m = float(input_param_config['Omega_m'])
        if 'z_lens' in input_param_config:
                emulated_z_lens = float(input_param_config['z_lens'])
        else:
                emulated_z_lens = 0.27 # default

        if 'z_source' in input_param_config:
                emulated_z_source = float(input_param_config['z_source'])
        else:
                emulated_z_source = 0.447 # default


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

        ## (3) emulate ln r_gm, convert to r_gm
        binmin_rgm, binmax_rgm, ln_rgm_model = emulate_bias_with_files(deriv_rgm_files,
                                                                    fiducial_rgm_file,
                                                                    fiducial_param_config,
                                                                    input_param_config)
        rgm_model = np.exp(ln_rgm_model)

        # sanity check
        tol=1.0e-5
        assert (((binmin_bg-binmin_bnl)<tol).all()) and (((binmax_bg-binmax_bnl)<tol).all())
        assert (((binmin_bg-binmin_rgm)<tol).all()) and (((binmax_bg-binmax_rgm)<tol).all())

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


        ## (5) compute xi_gm
        xi_gm = bg_model * rgm_model * ( bnl_model**2 * xi_mm_lin )

        ## (5) compute DS
        binmin_DS, binmax_DS, this_DS = compute_gamma_t(binmin_mm, binmax_mm, xi_gm, pimax=100.,
                                                        H0=100., Omega_m=emulated_Omega_m,
                                                        z_lens=emulated_z_lens,
                                                        z_source=emulated_z_source)


        ## (7) save model output
        np.savetxt(output_DS_file, np.c_[binmin_DS, binmax_DS, np.zeros(this_DS.shape[0]), this_DS], delimiter='\t')


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
#        parser.add_argument('emulator_settings_file')
        parser.add_argument('fiducial_param_file',help='*.template_param file describing fiducial parameters')
        parser.add_argument('fiducial_bg_file',help='fiducial b_g file')
        parser.add_argument('--deriv_bg_file',help='derivative file, param name for b_g',nargs=2,action='append',required=True)
        parser.add_argument('fiducial_bnl_file',help='fiducial b_nl file')
        parser.add_argument('--deriv_bnl_file',help='derivative file, param name for b_nl',nargs=2,action='append',required=True)
        parser.add_argument('fiducial_rgm_file',help='fiducial r_gm file')
        parser.add_argument('--deriv_rgm_file',help='derivative file, param name for r_gm',nargs=2,action='append',required=True)
        parser.add_argument('--xi_mm_linear',default=None,help='tabulated linear matter autocorrelation file')

        parser.add_argument('input_param_file',help='*.template_param file describing emulated parameters')
        parser.add_argument('output_DS_file',help='txt file output for emulated DS')

        args = parser.parse_args()
        
        emulate_DS_with_files(args.fiducial_bg_file, args.deriv_bg_file,
                              args.fiducial_bnl_file, args.deriv_bnl_file,
                              args.fiducial_rgm_file, args.deriv_rgm_file,
                              args.fiducial_param_file,
                              args.input_param_file,
                              args.output_DS_file,
                              args.xi_mm_linear)
