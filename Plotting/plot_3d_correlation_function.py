import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import scipy.interpolate
import argparse
from pathlib import Path

def plot_errorbar(ax, x, y, yerr=None, label=None, color=None, fmt=None):
        """plot errorbar with extended linestyles."""
        if fmt == '-..':
                ax.errorbar(x, y, yerr=yerr, label=label, color=color, dashes=[7,1.5,1,1.5,1,1.5])
        else:
                ax.errorbar(x, y, yerr=yerr, label=label, color=color, fmt=fmt)

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

def interpolate_or_nan(x,y):
        interpolator = scipy.interpolate.interp1d(x,y)
        xmin = np.min(x)
        xmax = np.max(x)
        def interp_fun(z):
                if z >= xmin and z <= xmax:
                        return interpolator(z)
                else:
                        return np.NaN
        return np.vectorize(interp_fun)

def plot_2pcf(input_files,logy=True,title=None,ylabel=None,residuals=True,
              residuals_only=False, linear_format_xaxis=True,ymin=None,ymax=None,xmin=None,xmax=None):
        color_cycle = ['black', 'blue', 'green', 'grey', 'grey', 'grey']
        style_cycle = ['-', '--', '-.', ':', '-..']

        axisfontsize=16
        legendfontsize=12
        fig = plt.figure(figsize=(4,4))

        if residuals:
                fig, ax_arr = plt.subplots(2, sharex=True)
                ax = ax_arr[0]
                ax_resid = ax_arr[1]
        elif not residuals_only:
                fig, ax = plt.subplots()
        else:
                fig, ax_resid = plt.subplots()

        global_binmin, global_binmax = [np.inf, -np.inf]
        global_fmin, global_fmax = [1.0, 1.0]
        global_min, global_max = [np.inf, -np.inf]
        first_bins = []
        first_corr = []
        for i, (f, input_label, cov_f) in enumerate(input_files):
                line_color = color_cycle[np.mod(i, len(color_cycle))]
                line_style = style_cycle[np.mod(i, len(style_cycle))]
                binmin, binmax, corr = load_correlation_file(f)
                if logy == True:
                        corr[corr <= 0.] = np.NaN
                err = None
                if cov_f is not '':
                        cov = np.loadtxt(cov_f)
                        err = np.sqrt(np.diag(cov))
                        #print(err)
                global_binmin = min(global_binmin, binmin[0])
                global_binmax = max(global_binmax, binmax[-1])
                bins = (np.array(binmax) + np.array(binmin))*0.5
                if i == 0:
                        first_bins = bins
                        first_corr = corr
                if not residuals_only:
                        global_min = min(global_min, np.nanmin(corr))
                        global_max = max(global_max, np.nanmax(corr))
                        plot_errorbar(ax, bins, corr, yerr=err, label=input_label, color=line_color, fmt=line_style)
                if residuals or residuals_only and np.allclose(bins,first_bins):
                        # interpolate corr onto first_bins
                        interp_corr = interpolate_or_nan(bins,corr)
                        f = interp_corr(first_bins)/first_corr
                        resid_err = None
                        if cov_f is not '':
                                interp_err = interpolate_or_nan(bins,err)
                                resid_err = interp_err(first_bins)/first_corr
                        #print(first_corr)
                        #print(resid_err)
                        global_fmin = min(global_fmin, np.nanmin(f))
                        global_fmax = max(global_fmax, np.nanmax(f))
                        plot_errorbar(ax_resid, first_bins, f, yerr=resid_err, label=input_label,
                                          color=line_color, fmt=line_style)

        if residuals:
                axes = [ax, ax_resid]
                resid_max = max(np.abs(1.0-global_fmin), np.abs(1.0-global_fmax))
#                resid_max = 0.2
                ax_resid.set_ylim((1.0-resid_max,1.0+resid_max))
                ax_resid.set_ylabel(r'ratio of %s' % ylabel,fontsize=axisfontsize)
                ax_resid.set_xlabel(r'$r$ ($h^{-1}$ Mpc)',fontsize=axisfontsize)
        elif residuals_only:
                axes = [ax_resid]
                resid_max = max(np.abs(1.0-global_fmin), np.abs(1.0-global_fmax))
                resid_max = 0.2
                ax_resid.set_ylim((1.0-resid_max,1.0+resid_max))
                ax_resid.set_ylabel(r'ratio of %s' % ylabel,fontsize=axisfontsize)
                ax_resid.set_xlabel(r'$r$ ($h^{-1}$ Mpc)',fontsize=axisfontsize)
        else:
                axes = [ax]
                ax.set_xlabel(r'$r$ ($h^{-1}$ Mpc)',fontsize=axisfontsize)
        
        for x in axes:
                x.set_xscale('log')
                if xmin == None and xmax == None:
                        x.set_xlim(add_whitespace_logscale(global_binmin, global_binmax))
                else:
                        x.set_xlim((xmin,xmax))
                x.xaxis.set_tick_params(top='on',direction='in',which='both',labelsize=12)
                x.yaxis.set_tick_params(right='on',direction='in',which='both',labelsize=12)
                if linear_format_xaxis:
                        x.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

        if logy and not residuals_only:
                ax.set_yscale('log')
        
        if not residuals_only:
                if logy:
                        yrange = add_whitespace_logscale(global_min,global_max)
                else:
                        yrange = add_whitespace(global_min,global_max)
                if ymin == None and ymax == None:
                        ax.set_ylim(yrange)
                else:
                        ax.set_ylim((ymin,ymax))
                ax.legend(loc='best',fontsize=legendfontsize)
                ax.set_ylabel(ylabel,fontsize=axisfontsize)
                ax.set_title(title)
        else:
                ax_resid.legend(loc='best',fontsize=legendfontsize)
                ax_resid.set_title(title)

        plt.tight_layout()

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--linear_yaxis',default=False,action='store_true',help='do not use a log scale for the y-axis')
        parser.add_argument('--log_format_xaxis',default=False,action='store_true',help='use a log-formatted scale for the x-axis')
        parser.add_argument('--residuals',default=False,action='store_true',help='plot ratio of inputs to first input')
        parser.add_argument('--residuals_only',default=False,action='store_true',help='only plot ratio of inputs to first input')
        parser.add_argument('--ymin',default=None,type=float)
        parser.add_argument('--ymax',default=None,type=float)
        parser.add_argument('--xmin',default=None,type=float)
        parser.add_argument('--xmax',default=None,type=float)
        parser.add_argument('output_file',help='pdf output for figure')
        parser.add_argument('figure_title',help='figure title')
        parser.add_argument('figure_yaxis',help='figure y-axis label')

        parser.add_argument('-f','--input_file',nargs=3,action='append',help='correlation function file')
        # this returns a list of tuples, one item for each input file
        # -- the first part of the tuple should be the filename
        # -- the second part of the tuple should be the plot label

        args = parser.parse_args()

        with PdfPages(args.output_file) as pdf:

                plot_2pcf(args.input_file, title=args.figure_title, ylabel=args.figure_yaxis,
                          logy=(not args.linear_yaxis), residuals=args.residuals,
                          residuals_only=args.residuals_only, linear_format_xaxis=(not args.log_format_xaxis),
                          ymin=args.ymin, ymax=args.ymax, xmin=args.xmin, xmax=args.xmax)
                pdf.savefig()
