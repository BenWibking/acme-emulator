import numpy as np
import matplotlib.pyplot as plt
import getdist
import getdist.plots
import argparse
from pylatexenc.latexencode import utf8tolatex
from pathlib import Path
from preliminize import preliminize
from plot_posterior import get_samples


def make_table_posteriors(posterior_samples_weights, filename=None):

	"""Generate LaTeX for table formatting to show the posterior parameter constraints."""

	table_header = r"""\begin{tabular}{l@{\qquad}ll@{\qquad}ll@{\qquad}ll@{\qquad}ll@{\qquad}ll@{\qquad}ll@{\qquad}ll}
\toprule
{Parameter} & {Fiducial} & {Lower $f_{\text{sat}}$} & {Higher $f_{\text{sat}}$} & {15\% Incompl.} & {`Baryons'} & \textbf{LOWZ} & $\bm{[> 2 \, h^{-1} \, \text{\textbf{Mpc}}]}$ \\
\midrule"""

	print_alensing = [True, False, False, False, True, True, True]

	table_footer = r"""\bottomrule
\end{tabular}"""

	posteriors_mysamples, hod_names_plot, cosmo_names_plot, cosmo_values_plot, labels_dict = \
									get_samples(posterior_samples_weights, set_labels=False)

	for j in range(len(posteriors_mysamples)):
		marge = posteriors_mysamples[j].getMargeStats()
		print(f"")
		
		for i, param_name in enumerate(cosmo_names_plot):
			tex = marge.texValues(getdist.types.NoLineTableFormatter(), param_name, limit=1)
			mean = marge.parWithName(param_name).mean
			limits = marge.parWithName(param_name).limits[0]
			print(f"{param_name}: {mean} {limits.lower} {limits.upper}")
			print(f"\t{tex}")
		print(f"")

	table_text = ""

	def add_text(this_text):
		nonlocal table_text
		table_text += this_text
	
	add_text(table_header + "\n")

	for i, param_name in enumerate(hod_names_plot):

		## new row of table
		add_text(labels_dict[param_name] + r" & ")

		for j in range(len(posteriors_mysamples)):

			## new column of table
			this_text = posteriors_mysamples[j].getLatex( param_name )[len(param_name):]
			if this_text.startswith(' ='):
				this_text = this_text[2:]

			add_text(r"$" + this_text + r"$" + r" & ")

		add_text(r" \\ " + "\n")

	add_text(r"\midrule" + "\n")

	for i, (param_name, param_true) in enumerate(zip(cosmo_names_plot, cosmo_values_plot)):

		## new row
		add_text(labels_dict[param_name] + r" & ")

		for j in range(len(posteriors_mysamples)):
			
			## new column
			if print_alensing[j] == False and param_name == 'Alensing':
				this_text = r"\text{N/A}"
			else:
				this_text = posteriors_mysamples[j].getLatex( param_name )[len(param_name):]
				if this_text.startswith(' ='):
					this_text = this_text[2:]

			add_text(r"$" + this_text + r"$" + r" & ")
		
		add_text(r" \\ " + "\n")

	add_text(table_footer)

	with open(filename, "w") as text_file:
		print(f"{table_text}", file=text_file)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--mcmc-chain', help='emcee chain output')
	parser.add_argument('--param-table', help='output LaTeX file', required=True)
	parser.add_argument('--multinest-dirs', nargs='*', help='multinest output directory')
	parser.add_argument('--labels', nargs='*')

	args = parser.parse_args()


	if args.mcmc_chain is not None:

		chain = np.loadtxt(args.mcmc_chain)
		burn_in_samples = 10000

		## plot posterior projections

		samples = chain[burn_in_samples:, 1:]
		make_table_posteriors(samples, filename=args.param_table)


	if args.multinest_dirs is not None:

		posteriors = []

		for multinest_dir, multinest_label in zip(args.multinest_dirs, args.labels):

			n_dims = 14
			multinest_samples = np.loadtxt(multinest_dir + '.txt')
			multinest_weights = multinest_samples[:, 0]
			multinest_lnL = multinest_samples[:, 1]
			multinest_params = multinest_samples[:, 2:2+n_dims]

			posteriors.append( (multinest_params, multinest_weights,
								utf8tolatex(multinest_label)) )

		make_table_posteriors(posteriors, filename=args.param_table)
