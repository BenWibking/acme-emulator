import numpy as np
import pandas as pd
import argparse
from pathlib import Path

def load_correlation_file(filename):
        table = np.loadtxt(filename,unpack=False)
        binmin, binmax, counts, corr = [table[:,i] for i in range(4)] 
        return binmin,binmax,corr

def table_2pcf(input_files, output_file, title=None, nrows=-1, do_ascii=False):
        """output LaTeX table for correlation functions as columns of a table"""
        first_binmin = []
        first_binmax = []
        ncols = len(input_files)
        mytable = []
        labels = []
        for i, (f, input_label) in enumerate(input_files):
                binmin, binmax, corr = load_correlation_file(f)
                bins = (np.array(binmax) + np.array(binmin))*0.5

                if i == 0:
                        first_binmin = binmin
                        first_binmax = binmax
                        nbins = binmin.shape[0]
                        mytable = np.zeros((nbins,ncols+2))
                        mytable[:,0] = binmin
                        mytable[:,1] = binmax
                else:
                        # make sure the binmin matches first_binmax, binmax also
                        assert(np.logical_and(np.allclose(binmin,first_binmin),
                                              np.allclose(binmax,first_binmax)))

                # add this file as a column to the table with header 'input_label'
                mytable[:,i+2] = corr
                labels.append(input_label)

        # generate the table
        tabledf = pd.DataFrame(mytable[:nrows])
        pd.set_option('display.max_colwidth', -1)
        if do_ascii == False:
                output_string = tabledf.to_latex(index=False, escape=False,
                                                 header=['$r_i$','$r_{i+1}$']+labels,
                                                 float_format='{:,.4f}'.format) #,column_format='lcc')
        else:
                output_string = tabledf.to_csv(index=False,
                                               header=['$r_i$','$r_{i+1}$']+labels)

        # save table to output_file
        with open(output_file,'w') as f:
                f.write(output_string)
        

parser = argparse.ArgumentParser()
parser.add_argument('output_file',help='pdf output for figure')
parser.add_argument('table_title',help='figure title')
parser.add_argument('--ascii',default=False,action='store_true')
parser.add_argument('--nrows',type=int,default=-1,help='number of rows of the table to print')
parser.add_argument('-f','--input_file',nargs=2,action='append',help='correlation function file')
# this returns a list of tuples, one item for each input file
# -- the first part of the tuple should be the filename
# -- the second part of the tuple should be the plot label

args = parser.parse_args()

table_2pcf(args.input_file, args.output_file, title=args.table_title, nrows=args.nrows,
           do_ascii=args.ascii)

