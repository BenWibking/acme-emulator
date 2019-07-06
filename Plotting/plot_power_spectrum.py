import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import argparse
from pathlib import Path

def load_spectrum_file(filename):
        print(filename)

        table = np.loadtxt(filename,unpack=False)
        k, pk = [table[:,i] for i in range(2)]

        from scipy.interpolate import interp1d
        P_interp = interp1d(k, pk)
        logkmin, logkmax = (np.log10(k[0]), np.log10(k[-1]))
        k_interp = np.logspace(logkmin, logkmax, 1024)
        k_interp[0] = k[0]
        k_interp[-1] = k[-1]

        return k_interp, P_interp(k_interp), P_interp

def plot_pk(input_files,logy=True,title=None,ylabel=None,residuals=True):
        if residuals:
                fig, ax_arr = plt.subplots(2, sharex=True)
                ax = ax_arr[0]
                ax_resid = ax_arr[1]
        else:
                fig, ax = plt.subplots()

        global_fmin, global_fmax = [np.inf, -np.inf]
        first_k = []
        first_pk = []
        first_pk_interp = []
        for i, (f, input_label) in enumerate(input_files):
                k, pk, pk_interp = load_spectrum_file(f)
                if i == 0:
                        first_k = k
                        first_pk = pk
                        first_pk_interp = pk_interp
                ax.plot(k, pk,'-',label=input_label)
                if residuals:
                        kmin, kmax = (max(first_k[0],k[0]), min(first_k[-1],k[-1]))
                        k_common = np.logspace(np.log10(kmin),np.log10(kmax),1024)
                        k_common[0] = kmin
                        k_common[-1] = kmax
                        f = pk_interp(k_common)/first_pk_interp(k_common)
                        global_fmin = min(global_fmin, np.min(f))
                        global_fmax = max(global_fmax, np.max(f))
                        ax_resid.plot(k, f, '-', label=input_label)

        if residuals:
                axes = [ax, ax_resid]
                resid_max = max(np.abs(1.0-global_fmin), np.abs(1.0-global_fmax))
                ax_resid.set_ylim((1.0-resid_max,1.0+resid_max))
                ax_resid.set_ylabel(r'ratio of %s' % ylabel)
                ax_resid.set_xlabel(r'wavenumber k ($h$ Mpc$^{-1}$)')
        else:
                axes = [ax]
                ax.set_xlabel(r'k ($h$ Mpc$^{-1}$)')
        
        for x in axes:
                x.set_xscale('log')

        ax.set_yscale('log')
        ax.set_ylabel(ylabel)
        ax.legend(loc='best')

        ax.set_title(title)
        plt.tight_layout()

parser = argparse.ArgumentParser()
parser.add_argument('--residuals',default=False,action='store_true',help='plot ratio of inputs to first input')
parser.add_argument('output_file',help='pdf output for figure')
parser.add_argument('figure_title',help='figure title')
parser.add_argument('figure_yaxis',help='figure y-axis label')
#parser.add_argument('input_files',nargs='*',help='correlation function files')
parser.add_argument('-f','--input_file',nargs=2,action='append',help='correlation function file')
# this returns a list of tuples, one item for each input file
# -- the first part of the tuple should be the filename
# -- the second part of the tuple should be the plot label

args = parser.parse_args()

with PdfPages(args.output_file) as pdf:
        plot_pk(args.input_file, title=args.figure_title, ylabel=args.figure_yaxis,
                residuals=args.residuals)
        pdf.savefig()
