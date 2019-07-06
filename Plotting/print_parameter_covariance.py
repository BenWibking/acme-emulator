#!/usr/bin/env python
import numpy as np
import argparse

def pretty_print_label(label):
        pretty_print = {}
        pretty_print['combined_om_s8'] = r'$\Delta \ln \sigma_8 \Omega_M^{0.3}$'
        pretty_print['sigma_8'] = r'$\Delta \ln \sigma_8$'
        pretty_print['siglogM'] = r'$\Delta \ln \sigma_{\log M}$'
        pretty_print['q_env'] = r'$\Delta Q_{env}$'
        pretty_print['ngal'] = r'$\Delta \ln n_{gal}$'
        pretty_print['ncen'] = r'$\Delta \ln n_{cen}$'
        pretty_print['alpha'] = r'$\Delta \ln \alpha$'
        pretty_print['M1_over_Mmin'] = r'$\Delta \ln \frac{M_1}{M_{min}}$'
        pretty_print['M0_over_M1'] = r'$\Delta \ln \frac{M_0}{M_1}$'
        pretty_print['H0'] = r'$\Delta \ln H_0$'
        pretty_print['Omega_M'] = r'$\Delta \ln \Omega_m$'
        pretty_print['del_gamma'] = r'$\Delta \gamma$'
        pretty_print['f_cen'] = r'$\Delta f_{cen}$'
        pretty_print['A_conc'] = r'$\Delta \ln A_{conc}$'
        pretty_print['delta_b'] = r'$\Delta B_{conc}$'
        pretty_print['delta_c'] = r'$\Delta C_{conc}$'
        return pretty_print[label]

def param_color(label):
        color = {}
        color['siglogM'] = 'green'
        color['ngal'] = 'green'
        color['ncen'] = 'green'
        color['alpha'] = 'green'
        color['M1_over_Mmin'] = 'green'
        color['M0_over_M1'] = 'green'
        color['q_env'] = 'blue'
        color['del_gamma'] = 'blue'
        color['A_conc'] = 'blue'
        color['delta_b'] = 'blue'
        color['delta_c'] = 'blue'
        color['f_cen'] = 'blue'
        color['H0'] = 'black'
        color['Omega_M'] = 'black'
        color['sigma_8'] = 'black'
        color['combined_om_s8'] = 'black'
        return color[label]

def load_covariance_file(filename):
	table = np.loadtxt(filename)
	return table

def diagnose_condition(matrix):
	# check condition number -- if large (>10e6), this indicates something went wrong
	# rule of thumb: log10(condition) gives the number of decimal digits of precision loss
	# if >6 for single precision, all precision is lost; if >15 for double precision, all precision is lost
	#   (i.e. answers are completely meaningless)
	condition = np.linalg.cond(matrix)
	print("")
	if condition>1e6:
		print("WARNING: bad condition number! these results are probably wrong!")
	print("condition number of covariance matrix:", condition)
	bits_lost = np.log2(condition)
	single_bits_remaining = 32.0-bits_lost
	double_bits_remaining = 64.0-bits_lost
	single_digits_remaining = 6.0-np.log10(condition)
	double_digits_remaining = 15.0-np.log10(condition)
	if double_bits_remaining>0.:
		print("estimated decimal places of precision for double precision computations:", double_digits_remaining)
	else:
		print("WARNING: double-precision computations with this matrix are meaningless!")
	print("")

def print_covariance(input_files, output_file):
        tick_angle = 90.
        plot_colors = ['blue', 'red', 'black']

        cosmo_params = ['sigma_8', 'Omega_M']

        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        fig = plt.figure(figsize=(10,10))
        set_of_file_data = []

        # N.B. input files must have the same dimensions
        for input_file in input_files:
                print("")
                print("INPUT FILE:",input_file)
                print("")

                cov = load_covariance_file(input_file)
                with open(input_file,'r') as f:
                        line = f.readline().strip()
                params = line[1:].strip().split(' ')
                print("*** (Fisher-matrix-estimated) marginalized uncertainties on parameters: ***")
                # compute condition number and output diagnostic info
                diagnose_condition(cov)

                p = []
                sigma_z_sq = []

                param_limits = [(-2*np.sqrt(cov[i,i]),2*np.sqrt(cov[i,i])) for i in range(cov.shape[0])] # 2-sigma contours

                # compute ellipses
                # reference: https://arxiv.org/pdf/0906.4123.pdf
                print("*** uncertainty ellipses:")
                print("*** (param_x) (param_y) (correlation) (slope)")
                ellipses = []
                for i in range(cov.shape[0]):
                        for j in range(i):
                                xparam = params[j]
                                yparam = params[i]
                                sigma_x_sq = cov[j,j]
                                sigma_y_sq = cov[i,i]
                                sigma_xy = cov[i,j]
                                term1 = (sigma_x_sq + sigma_y_sq)/2.0 
                                term2 = np.sqrt( ((sigma_x_sq - sigma_y_sq)**2)/4.0 + sigma_xy**2 )
                                a = np.sqrt( term1 + term2 )
                                b = np.sqrt( term1 - term2 )
                                angle_radians = np.arctan2(2.0*sigma_xy, sigma_x_sq - sigma_y_sq)/2.0
                                angle_deg = angle_radians * 180. / np.pi
                                slope = np.tan(angle_radians)
                                ellipses.append((i,j,xparam,yparam,a,b,angle_deg))
                                corr = cov[i,j] / np.sqrt(cov[i,i]*cov[j,j])
                                if xparam in cosmo_params and yparam in cosmo_params:
                                        if xparam == 'Omega_M' and yparam == 'sigma_8':
                                                sigma_om_sq = sigma_x_sq
                                                sigma_s8_sq = sigma_y_sq
                                        elif xparam == 'sigma_8' and yparam == 'Omega_M':
                                                sigma_s8_sq = sigma_x_sq
                                                sigma_om_sq = sigma_y_sq
                                        p = -sigma_xy / sigma_om_sq
                                        sigma_z_sq = (p**2)*sigma_om_sq + sigma_s8_sq + 2.0*p*sigma_xy

                                print(xparam,'\t',yparam,'\t',sigma_x_sq,'\t',sigma_y_sq,'\t',sigma_xy)#'\t',corr,'\t',slope)
                print("")

                plot_color = plot_colors.pop()
                set_of_file_data.append((ellipses,cov,param_limits,plot_color))
                
                print("*** marginalized parameter uncertainties")
                for i in range(cov.shape[0]):
                        print(params[i], '\t\t', np.sqrt(cov[i,i])) # print sigma_i
                print("")
                print('best constrained parameter: sigma_8 * Om**%s' % p)
                print('\tuncertainty: %s' % np.sqrt(sigma_z_sq))
                print("")

        # ensure that all input files have the same dimensions
        dims = set_of_file_data[0][1].shape
        for data in set_of_file_data[1:]:
                ellipses,cov,param_limits,plot_color = data
                if(cov.shape != dims):
                        raise Exception("input files do not have the same dimensions!")

        # create subplots
        ellipses,cov,param_limits,plot_color = set_of_file_data[0]
        subplots = []
        for i in range(cov.shape[0]):
                row = []
                for j in range(i+1):
                        ax = plt.subplot2grid((cov.shape[0],cov.shape[0]),(i,j),aspect='equal')
                        plt.xticks(rotation=tick_angle)
                        plt.tick_params(axis='both',which='major',labelsize=10)
                        plt.tick_params(axis='both',which='minor',labelsize=8)
                        ax.set_aspect('auto')
                        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
                        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
                        row.append(ax)
                subplots.append(row)

        xlim = np.zeros(cov.shape)
        ylim = np.zeros(cov.shape)
        max_amp = np.zeros(cov.shape[0])
        for data in set_of_file_data:
                ellipses,cov,param_limits,plot_color = data
                for i in range(cov.shape[0]):
                        this_amp = 1.0/(np.sqrt(2.0*np.pi*cov[i,i]))
                        max_amp[i] = max(this_amp, max_amp[i])
                        for j in range(i+1):
                                this_xlim = 2.0*np.sqrt(cov[j,j])
                                this_ylim = 2.0*np.sqrt(cov[i,i])
                                xlim[i,j] = max(this_xlim, xlim[i,j])
                                ylim[i,j] = max(this_ylim, ylim[i,j])

        # plot ellipses (non-diagonal parameter covariances)
        for file_data in set_of_file_data:
                ellipses,cov,param_limits,plot_color = file_data # unpack file_data
                for i,j,xlabel,ylabel,a,b,angle_deg in ellipses:
                        ax = subplots[i][j]
                        scales = [(2.48,'red'), (1.52,'orange')] # 95%, 68% confidence ellipses
                        for scale,face_color in scales:
                                e = Ellipse(xy=[0.,0.], width=a*scale, height=b*scale, angle=angle_deg,
                                            color=plot_color)
                                e.set_facecolor(face_color)
                                e.set_alpha(0.5)
                                ax.add_patch(e)
                        ax.set_xlim(-xlim[i,j],xlim[i,j])
                        ax.set_ylim(-ylim[i,j],ylim[i,j])
                        if j > 0:
                                ax.yaxis.set_ticklabels([])
                                ax.set_yticks([])
                        else:
                                ax.set_ylabel(pretty_print_label(ylabel),
                                              color=param_color(ylabel))
                                ax.yaxis.set_label_coords(-0.5, 0.5)
                                ax.yaxis.tick_left()

                        if i < cov.shape[0]-1:
                                ax.xaxis.set_ticklabels([])
                                ax.set_xticks([])
                        else:
                                ax.set_xlabel(pretty_print_label(xlabel),
                                              color=param_color(xlabel))
                                ax.xaxis.set_label_coords(0.5,-0.5)
                                ax.xaxis.tick_bottom()

        # plot diagonal elements
        for file_data in set_of_file_data:
                ellipses,cov,param_limits,plot_color = file_data # unpack file_data
                for i in range(cov.shape[0]):
                        ax = subplots[i][i]
                        sigma = np.sqrt(cov[i,i])
                        x = np.linspace(-4.0*sigma, 4.0*sigma, 100)
                        fun = np.exp(-0.5*(x/sigma)**2) / (sigma * np.sqrt(2.0*np.pi))
                        ax.plot(x, fun, color=plot_color)
                        ax.set_title('$\sigma = %.3f$' % sigma,
                                     color=param_color(params[i]))
                        ax.set_xlim(-xlim[i,i],xlim[i,i])
                        ax.set_ylim(0., max_amp[i])
                        ax.yaxis.set_ticklabels([])
                        ax.set_yticks([])
                        ax.yaxis.tick_left()

                        if i == cov.shape[0]-1:
                                ax.set_xlabel(pretty_print_label(params[i]),
                                              color=param_color(params[i]))
                                ax.xaxis.set_label_coords(0.5,-0.5)
                                ax.xaxis.tick_bottom()
                        else:
                                ax.xaxis.set_ticklabels([])
                                ax.set_xticks([])

        #fig.tight_layout(pad=.8)
        fig.subplots_adjust(wspace=0.,hspace=0.)
        #plt.suptitle(r"Emulator forecast for galaxy-galaxy lensing+clustering")
        plt.savefig(output_file)

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('output_file')
        parser.add_argument('input_cov_files',nargs='*',help='txt file input for inverse Fisher matrix')
        args = parser.parse_args()
        
        print_covariance(args.input_cov_files, args.output_file)
