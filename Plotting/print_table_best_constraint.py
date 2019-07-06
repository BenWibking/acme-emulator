#!/usr/bin/env python
import numpy as np
import argparse
from print_parameter_covariance import pretty_print_label, param_color
from plot_cumulative_parameter_constraints import load_covariance_file, param_index

def plot_parameter_constraints(input_files, output_file2):
        set_of_file_data = []
        for input_file,label in input_files:
                cov, params = load_covariance_file(input_file)
                set_of_file_data.append((cov,params, label))

        # ensure that all input files have the same dimensions
        dims = set_of_file_data[0][0].shape
        params = set_of_file_data[0][1]
        # for data in set_of_file_data[1:]:
        #         cov, included_params, label = data
        #         if(cov.shape != dims):
        #                 raise Exception("input files do not have the same dimensions!")
        #         if(included_params != params):
        #                 raise Exception("input files do not have the same parameters!")

        import pandas as pd
        labels = []
        for i,data in enumerate(set_of_file_data):
                cov, params, label = data
                labels.append(label)

        # compute best-constrained combination of Om**p * s8 (put in separate table)
        combined_param_table = np.zeros((len(input_files),2))
        for i,file_data in enumerate(set_of_file_data):
                cov, params, label = file_data
                for j, param in enumerate(params):
                        if param == 'Omega_M':
                                om_idx = j
                        elif param == 'sigma_8':
                                s8_idx = j
                
                sigma_om_sq = cov[om_idx, om_idx]
                sigma_s8_sq = cov[s8_idx, s8_idx]
                sigma_xy = cov[om_idx, s8_idx]
                
                p = -sigma_xy / sigma_om_sq
                sigma_z_sq = (p**2)*sigma_om_sq + sigma_s8_sq + 2.0*p*sigma_xy
        
                combined_param_table[i,0] = p
                combined_param_table[i,1] = np.sqrt(sigma_z_sq)

        combined_param_tabledf = pd.DataFrame(combined_param_table,index=labels)
        pd.set_option('display.max_colwidth', -1)
        latex_combined = combined_param_tabledf.to_latex(index=True, escape=False,
                                                         header=[r'$p$',r'best-constrained $\sigma_8 \Omega_m^p$'],
                                                         float_format='{:,.3f}'.format,column_format='lcc')

        with open(output_file2,'w') as f:
                f.write(latex_combined)


parser = argparse.ArgumentParser()
parser.add_argument('output_file2')
parser.add_argument('-f','--input_cov_files',nargs=2,action='append',help='txt file input for inverse Fisher matrix')
args = parser.parse_args()

plot_parameter_constraints(args.input_cov_files, args.output_file2)
