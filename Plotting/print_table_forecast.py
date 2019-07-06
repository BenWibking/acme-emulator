#!/usr/bin/env python
import numpy as np
import argparse
from print_parameter_covariance import param_color
from plot_cumulative_parameter_constraints import load_covariance_file, param_index

def pretty_print_label(label):
        pretty_print = {}
        pretty_print['combined_om_s8'] = r'$\Delta \ln \sigma_8 \Omega_M^{0.3}$'
        pretty_print['sigma_8'] = r'$\Delta \ln \sigma_8$'
        pretty_print['siglogM'] = r'$\Delta \ln \sigma_{\log M}$'
        pretty_print['q_env'] = r'$\Delta Q_{\textrm{env}}$'
        pretty_print['ngal'] = r'$\Delta \ln n_{\textrm{gal}}$'
        pretty_print['ncen'] = r'$\Delta \ln n_{\textrm{cen}}$'
        pretty_print['alpha'] = r'$\Delta \ln \alpha$'
        pretty_print['M1_over_Mmin'] = r'$\Delta \ln \frac{M_1}{M_{\textrm{min}}}$'
        pretty_print['H0'] = r'$\Delta \ln H_0$'
        pretty_print['Omega_M'] = r'$\Delta \ln \Omega_m$'
        pretty_print['del_gamma'] = r'$\Delta \gamma$'
        pretty_print['f_cen'] = r'$\Delta f_{\textrm{cen}}$'
        return pretty_print[label]

def plot_parameter_constraints(input_files, output_file1, col_file):
        set_of_file_data = []
        for input_file,label in input_files:
                cov, this_params = load_covariance_file(input_file)
                set_of_file_data.append((cov,this_params, label))

        print("load column file: {}".format(col_file))
        _, table_params = load_covariance_file(col_file) # get table columns from this file
        print(table_params)

        # ensure that all input files have the same dimensions
        #dims = set_of_file_data[0][0].shape
        #params = set_of_file_data[0][1]
        # for data in set_of_file_data[1:]:
        #         cov, included_params, label = data
        #         if(cov.shape != dims):
        #                 raise Exception("input files do not have the same dimensions!")
        #         if(included_params != params):
        #                 raise Exception("input files do not have the same parameters!")
        
        def is_param(x,y):
                if x==y:
                        return True
                elif x=='ngal' and y=='ncen':
                        return True
                else:
                        return False

        table = np.zeros((len(input_files),len(table_params)))
        table.fill(np.NaN) # allow for missing data, will be replaced with '---' in table
        for i,file_data in enumerate(set_of_file_data):
                cov, this_params, label = file_data
                for j,param in enumerate(this_params):
                        sigma = np.sqrt(cov[j,j])
                        # find index in 'table_params' for parameter 'param'
                        idx = [q for q,p in enumerate(table_params) if is_param(p,param)]
                        table[i,idx] = sigma

        import pandas as pd
        labels = []
        for i,data in enumerate(set_of_file_data):
                cov, this_params, label = data
                labels.append(label)
        tabledf = pd.DataFrame(table,index=labels)
        pd.set_option('display.max_colwidth', -1)
        latex = tabledf.to_latex(index=True, escape=False, header=[pretty_print_label(p) for p in table_params],float_format='{:,.3f}'.format,column_format='lccccccccc')

        # replace 'nan' in latex with '---'
        latex = latex.replace('nan','---')

        with open(output_file1,'w') as f:
#                f.write(r"""\documentclass{article}
#\usepackage[margin=0.5in]{geometry}
#\usepackage{booktabs}
#\begin{document}
#\begin{center}""")
                f.write(latex)
#                f.write(r"""\end{center}
#\end{document}""")


parser = argparse.ArgumentParser()
parser.add_argument('output_file1')
parser.add_argument('column_file')
parser.add_argument('-f','--input_cov_files',nargs=2,action='append',help='txt file input for inverse Fisher matrix')
args = parser.parse_args()

plot_parameter_constraints(args.input_cov_files, args.output_file1, args.column_file)
