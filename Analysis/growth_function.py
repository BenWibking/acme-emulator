#!/usr/bin/env python
import numpy as np
import camb

params = camb.model.CAMBparams()
params.set_cosmology(H0=70.26,ombh2=0.022,omch2=0.12,omk=0.0)
params.set_dark_energy(w=-1.0)
params.InitPower.set_params(ns=0.96)
params.NonLinear = camb.model.NonLinear_none
params.set_matter_power(redshifts=[0.1, 0.0])

model = camb.get_results(params)
sigma_8_array = model.get_sigma8()

growth_factor = sigma_8_array[0] / sigma_8_array[1]
print(growth_factor)
