import numpy as np
import numpy.linalg as linalg
from numba import jit
from math import exp
import time
import matplotlib.pyplot as plt

from train_emu import read_data, model_data, compute_labels
from plot_emulator import tex_escape

def test_data(sims_dir, redshift_dir, param_files, plot_filename, input_emu_filename):
    """read in test data"""
    x, y_allbins, binmin, binmax = read_data(sims_dir, redshift_dir, param_files,
        filename_ext='.weighted_wp.txt')
    binmed = 0.5*(binmin+binmax)
    test_X = compute_labels(x) # can modify zero-point, take logarithms, etc.

    """ read in emulator """
    import h5py
    f = h5py.File(input_emu_filename, 'r')
    training_X = f['input_labels'][:]
    coefs = f['coefs'][:]
    kernel_hypers = f['kernel_hypers'][:]
    y_mean = f['mean_y'][:]
    y_sigma = f['sigma_y'][:]

    # do this for each radial bin
    rms_residuals = np.zeros(y_allbins.shape[0])
    frac_rms_residuals = np.zeros(y_allbins.shape[0])
    for j in range(y_allbins.shape[0]):
        test_y = y_allbins[j,:] # test data

        training_y0 = y_mean[j]
        training_sigma_y = y_sigma[j]
        c = coefs[j,:]
        h = kernel_hypers[j,:]
        y_model_x = training_sigma_y*model_data(training_X, test_X, c, h) + training_y0

        rms_residuals[j]= np.sqrt(np.mean((y_model_x - test_y)**2))
        frac_rms_residuals[j] = rms_residuals[j] / y_mean[j]

        print('[bin {}] frac rms test residuals = {}'.format(j,frac_rms_residuals[j]))

        do_plot = False
        #if frac_rms_residuals[j] > 1.0:
        #    do_plot = True

        ## plot model
        if do_plot == True:
            print('[bin {}] plotting residuals...'.format(j),end='',flush=True)
            fig = plt.figure()
            plt.scatter(x[:,0],y,label='data')
            plt.scatter(x[:,0],y_model_x,label='model',marker='+')
            plt.ylabel('$f(x)$')
            plt.xlabel('x[0]')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig('model_fit_bin_{}.pdf'.format(j))
            plt.close(fig)

            fig = plt.figure()
            plt.scatter(x[:,0], y_model_x - y,label='model residuals')
            plt.ylabel('error in $f(x)$')
            plt.xlabel('x')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig('model_fit_residuals_bin_{}.pdf'.format(j))
            plt.close(fig)
            print('done.\n')

        print('')

    ## plot cross-validation results
    plt.figure()
    plt.scatter(binmed,frac_rms_residuals, label='rms test residuals / mean signal')
#    plt.scatter(binmed,rms_residuals, label='rms model residuals / mean signal')

    plt.xscale('log')
    plt.yscale('log')
    #minplot = min(np.min(frac_rms_looe),np.min(frac_rms_residuals),np.min(frac_rms_kfold))
    minplot = 1e-4
    maxplot = 1e-0
    plt.ylim((minplot,maxplot))
    plt.xlim((binmin[0],binmax[-1]))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.title(tex_escape(plot_filename))
    plt.savefig(plot_filename)
    plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('plot_filename')
    parser.add_argument('input_emu_filename')
    parser.add_argument('sims_dir')
    parser.add_argument('redshift_dir')
    parser.add_argument('param_files',nargs='*')
    args = parser.parse_args()
    test_data(args.sims_dir, args.redshift_dir, args.param_files,
              args.plot_filename, args.input_emu_filename)
