import numpy as np


N = 200
N_params = 5

import diversipy.hycusampling as lhs
lhc = lhs.improved_lhd_matrix(N, N_params)
samples = lhs.edge_lhs(lhc)

for idx in range(samples.shape[0]):
    delta_param = samples[idx,:]
    print(delta_param)
