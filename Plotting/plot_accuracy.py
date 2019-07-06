import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import argparse
from pathlib import Path

def plot_errorbar(ax, x, y, yerr=None, label=None, color=None, fmt=None, alpha=1.0):

		"""plot errorbar with extended linestyles."""

		if fmt == '-..':
				ax.errorbar(x, y, yerr=yerr, label=label, color=color, dashes=[7,1.5,1,1.5,1,1.5], alpha=alpha)
		else:
				ax.errorbar(x, y, yerr=yerr, label=label, color=color, fmt=fmt, alpha=alpha)


def add_whitespace(global_min, global_max, margin_fraction=0.05):

		"""add/subtract 5% of |global_max - global_min| to global_min,global_max"""

		this_min = global_min
		this_max = global_max

		global_range = np.abs(this_max - this_min)
		margin = margin_fraction * global_range

		result = (this_min - margin, this_max + margin)
		return result


def add_whitespace_logscale(global_min, global_max, margin_fraction=0.05):

		"""add/subtract 5% of |global_max - global_min| to global_min,global_max"""

		this_min = np.log10(global_min)
		this_max = np.log10(global_max)

		global_range = np.abs(this_max - this_min)
		margin = margin_fraction * global_range

		result = (10.**(this_min-margin), 10.**(this_max+margin))
		return result


def load_correlation_file(filename):

		table = np.loadtxt(filename,unpack=False)
		binmin, binmax, counts, corr = [table[:,i] for i in range(4)]                        
		return binmin,binmax,corr


def plot_2pcf(input_files,title=None,ylabel=None,linear_format_xaxis=True,linear_format_yaxis=True,clip_max=None):

		color_cycle = ['black']
		style_cycle = ['--']

		axisfontsize=16
		legendfontsize=12
		fig = plt.figure(figsize=(4,4))
		fig, ax = plt.subplots()

		global_binmin, global_binmax = [np.inf, -np.inf]
		global_min, global_max = [np.inf, -np.inf]

		corr_array = []
		rms_list = []
		for i, f in enumerate(input_files):
				line_color = color_cycle[np.mod(i, len(color_cycle))]
				line_style = style_cycle[np.mod(i, len(style_cycle))]
				binmin, binmax, corr = load_correlation_file(f)
				if corr_array == []:
						corr_array = np.zeros((len(input_files), corr.shape[0]))
				corr_array[i,:] = corr

				global_binmin = min(global_binmin, binmin[0])
				global_binmax = max(global_binmax, binmax[-1])
				bins = (np.array(binmax) + np.array(binmin))*0.5

				global_min = min(global_min, np.nanmin(corr))
				global_max = max(global_max, np.nanmax(corr))
				plot_errorbar(ax, bins, corr, yerr=None, label=None, color=line_color, fmt=line_style, alpha=0.2)

				# compute mean squared error
				rms = np.sqrt(np.mean(corr**2))
				rms_list.append((f,rms))
				
		print("")

		## compute dispersion

		disp_error = np.std(corr_array,axis=0)
		mean_signal = np.mean(corr_array,axis=0)
		upper_disp_err = mean_signal + disp_error
		lower_disp_err = mean_signal - disp_error

#		log_mean = np.log10(mean_signal)
#		upper_disp_err = 10.0**( log_mean + np.std(np.log10(corr_array), axis=0) )
#		lower_disp_err = 10.0**( log_mean - np.std(np.log10(corr_array), axis=0) )

		ax.fill_between(bins, lower_disp_err, upper_disp_err,
						label='mean + dispersion', color='gray', zorder=20)
		ax.plot(bins, mean_signal, '--', color='black', zorder=21)

		rms_list.sort(key=lambda x: x[1], reverse=True)
#        for f, rms in rms_list:
#                print('file: {} rms = {}'.format(f,rms))

		ax.set_xlabel(r'$r_p$ [$h^{-1}$ Mpc]',fontsize=axisfontsize)
		ax.set_xscale('log')

		if not linear_format_yaxis:
				ax.set_yscale('log')

		ax.set_xlim(add_whitespace_logscale(global_binmin, global_binmax))
		ax.xaxis.set_tick_params(top=True,direction='in',which='both',labelsize=12)
		ax.yaxis.set_tick_params(right=True,direction='in',which='both',labelsize=12)
		if linear_format_xaxis:
				ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

		if clip_max != None:
				global_min, global_max = [-np.abs(clip_max), np.abs(clip_max)]

		yrange = add_whitespace(global_min,global_max)
		ax.set_ylim(yrange)
		ax.legend(loc='best',fontsize=legendfontsize)
		ax.set_ylabel(ylabel,fontsize=axisfontsize)
		ax.set_title(title)

		plt.tight_layout()


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	
	parser.add_argument('--log-format-xaxis',default=False,action='store_true',
						help='use a log-formatted scale for the x-axis')
	parser.add_argument('--log-format-yaxis',default=False,action='store_true',
						help='use a log-formatted scale for the y-axis')
	parser.add_argument('--clip-max',default=None,type=float)
	
	parser.add_argument('output_file',help='pdf output for figure')
	parser.add_argument('figure_title',help='figure title')
	parser.add_argument('figure_yaxis',help='figure y-axis label')
	parser.add_argument('input_files',nargs='*',help='correlation function files')
	
	args = parser.parse_args()
	
	with PdfPages(args.output_file) as pdf:
			plot_2pcf(args.input_files,
					title=args.figure_title,
					ylabel=args.figure_yaxis,
					linear_format_xaxis=(not args.log_format_xaxis),
					  linear_format_yaxis=(not args.log_format_yaxis),
					clip_max=args.clip_max)
			pdf.savefig()
