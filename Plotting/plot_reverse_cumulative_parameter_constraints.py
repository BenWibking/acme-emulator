#!/usr/bin/env python
import numpy as np
import numpy.linalg as linalg
import argparse
from print_parameter_covariance import param_color, pretty_print_label

# def pretty_print_label(label):
#         pretty_print = {}
#         pretty_print['sigma_8'] = r'$\sigma_8$'
#         pretty_print['siglogM'] = r'$\sigma_{\log M}$'
#         pretty_print['q_env'] = r'$Q_{env}$'
#         pretty_print['ngal'] = r'$n_{gal}$'
#         pretty_print['alpha'] = r'$\alpha$'
#         pretty_print['M1_over_Mmin'] = r'$\frac{M_1}{M_{min}}$'
#         pretty_print['H0'] = r'$H_0$'
#         pretty_print['Omega_M'] = r'$\Omega_m$'
#         pretty_print['del_gamma'] = r'$\Delta \gamma$'
#         return pretty_print[label]

def load_covariance_file(filename):
        table = np.loadtxt(filename)

        # read header
        with open(filename,'r') as f:
                line = f.readline().strip()
        params = line[1:].strip().split(' ')

        return table, params

def restrict_cov(cov, params, include_params):
        # invert
        F = linalg.inv(cov)

        # delete rows/columns not in include_params
        mask = [param in include_params for param in params]
        F_restrict = F[mask,:][:,mask]
        params_restrict = [param for param in params if param in include_params]

        # invert again
        cov_restrict = linalg.inv(F_restrict)

        return cov_restrict, params_restrict

def param_index(param, params):
        index = None
        for i in range(len(params)):
                if param == params[i]:
                        index = i
                        break
        return index

def find_next_param(cov, included_params=[], params=[], first_param=None):
        """subtract one element from included_params and return the element
        that most reduces the marginalized uncertainty on first_param."""
        best_param = []
        best_sigma = np.inf
        # we don't want to remove 'Omega_M' or 'sigma_8'
        trial_params = [param for param in included_params if param != 'Omega_M' and param != 'sigma_8']
        for param in trial_params:
                new_included_params = list(set(included_params) - set([param]))
                cov_restrict, params_restrict = restrict_cov(cov, params, new_included_params)
                i = param_index(first_param, params_restrict)
                sigma = np.sqrt(cov_restrict[i,i])
                print(param,sigma)
                if sigma < best_sigma:
                        best_param = param
                        best_sigma = sigma
        return best_param, best_sigma

def set_complement(subset, universe):
        """returns complement of subset in universe"""
        c = []
        for elem in universe:
                if elem not in subset:
                        c.append(elem)
        return c

def compute_sigma_restricted(cov, included_params=[], params=[], first_param=None):
        cov_restrict, params_restrict = restrict_cov(cov,params,included_params)
        i = param_index(first_param,params_restrict)
        sigma = np.sqrt(cov_restrict[i,i])
        return sigma

def plot_parameter_constraints(input_files, output_file):
        color_cycle = ['black', 'blue', 'green', 'red', 'orange']
        style_cycle = ['-o', '--o', '-.o', ':o', '-o']
        mfc_cycle = [None, None, None, None, 'none']

        set_of_file_data = []

        for input_file, label in input_files:
                cov, params = load_covariance_file(input_file)
                set_of_file_data.append((cov,params,label))

        # ensure that all input files have the same dimensions
        dims = set_of_file_data[0][0].shape
        for data in set_of_file_data[1:]:
                cov, params, label = data
                if(cov.shape != dims):
                        raise Exception("input files do not have the same dimensions!")

        file_results = []
        excluded_params = [] # this will be the same for all input files, but determine by file[0]
        for file_data in set_of_file_data:
                cov, params, label = file_data
                uncertainties_chain = []
                first_param = 'sigma_8'
                second_param = 'Omega_M'
                ## included_params are the params we marginalize over, excluded_params are fixed

                if excluded_params == []: # i.e. this is the first input file
                        # start with just sigma_8
                        excluded_params = [second_param]
                        sigma = compute_sigma_restricted(cov, included_params=set_complement(excluded_params,params), params=params, first_param=first_param)
                        print('first_param: %s, first_sigma: %s' % (first_param, sigma))
                        uncertainties_chain.append(sigma)

                        # add additional parameters
                        while len(excluded_params) < len(params)-1:
                                next_param, next_sigma = find_next_param(cov, included_params=set_complement(excluded_params,params), first_param=first_param, params=params)
                                print("next_param: %s, next_sigma %s\n" % (next_param,next_sigma))
                                excluded_params.append(next_param)
                                uncertainties_chain.append(next_sigma)

                        file_results.append((uncertainties_chain, excluded_params, label))
                else: # i.e. this is not the first input file
                        # use already-computed included_params
                        this_excluded_params = []     
                        for param in excluded_params:
                                this_excluded_params.append(param)
                                sigma = compute_sigma_restricted(cov, included_params=set_complement(this_excluded_params,params), params=params, first_param=first_param)
                                uncertainties_chain.append(sigma)
                        file_results.append((uncertainties_chain, excluded_params, label))

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(5,5))
        ax = plt.subplot(111)
        for i,file_result in enumerate(file_results):
                line_color = color_cycle[np.mod(i,len(color_cycle))]
                line_style = style_cycle[np.mod(i,len(style_cycle))]
                mfc_style = mfc_cycle[np.mod(i,len(mfc_cycle))]
                param_uncertainty, params, label = file_result
                print(params)
                ranks = np.linspace(1., float(len(param_uncertainty)), len(param_uncertainty))
                ax.plot(ranks, param_uncertainty, line_style, label=label, color=line_color,
                        mfc=mfc_style)
                ax.set_xticks(ranks)
                ax.xaxis.set_ticklabels([pretty_print_label(param) for param in params])
                ax.set_xlim(ranks[0]-0.5, ranks[-1]+0.5)
        ax.legend(loc='best')
        ax.set_ylabel('marginalized fractional uncertainty in $\sigma_8$')
        ax.set_xlabel('cumulatively (from the left) fixed parameters')
        ax.xaxis.tick_bottom()
        #ax.yaxis.tick_left()
        ax.yaxis.set_tick_params(right='on',direction='in',which='both')
        ax.set_yscale('log')

        plt.savefig(output_file)

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('output_file')
        parser.add_argument('-f','--input_cov_files',nargs=2,action='append',help='txt file input for inverse Fisher matrix')
        args = parser.parse_args()

        plot_parameter_constraints(args.input_cov_files, args.output_file)
