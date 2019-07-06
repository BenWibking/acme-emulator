import numpy as np
import scipy.linalg
import argparse
from pathlib import Path


if __name__ == '__main__':
	
	"""compute the eigenvalues of the given covariance matrix.
		Plot the noise threshold = \sqrt{ 2 / N_mocks }."""
		
	parser = argparse.ArgumentParser()
	parser.add_argument('cov_matrix')
	parser.add_argument('--nmocks', type=float, required=True)
	args = parser.parse_args()
	
	
	## compute
	
	cov = np.loadtxt(args.cov_matrix)
	print(f"cov.shape = {cov.shape}")
	print(f"Nmocks = {args.nmocks}")
	
	threshold = np.sqrt( 2.0 / args.nmocks )
	print(f"noise threshold = {threshold}")
	
	U, s, V = scipy.linalg.svd(cov)
	eval, evec = scipy.linalg.eigh(cov)
	eval = eval[::-1]

	# the singular values and the eigenvalues are equal if it's a symmetric real matrix
	assert( np.allclose( s, eval ) )
	
	print(f"eigenvalues: {eval}")
	
	
	## plot
	
	import matplotlib.pyplot as plt
	from matplotlib import rcParams
	rcParams['text.latex.preamble'] = r"""
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
"""
	plt.figure()
	
	plt.scatter(range(1,len(eval)+1), eval, label='eigenvalues')
	plt.plot(range(1,len(eval)+1), eval)
	plt.axhline(threshold,
				linestyle='--', color='black', label=r'noise threshold $\gtrsim \sqrt{2 / N_{mocks}}$')
				
	plt.yscale('log')
	plt.xlabel('eigenvector \#')
	plt.ylabel('eigenvalue')
	plt.legend(loc='best')
	
	plt.tight_layout()
	plt.savefig( str(Path(args.cov_matrix).with_suffix('.eigenvalues.pdf')) )
	