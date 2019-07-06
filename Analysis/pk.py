#!/usr/bin/env python
import numpy as np
from scipy.integrate import romb as romberg

# sigma^2 integral is (1/(2 pi^2)) k^3 P(k) ((3 sin(kR) - kR cos(kR))/(kR)^3)^2 d(ln k)
# P is log-sampled
npoints = (1024)+1
k_log_space = np.logspace(-3, 2, npoints)
dlogk = np.log(k_log_space[1]) - np.log(k_log_space[0])
P_log_space = 1.0 * k_log_space**1.0

R = 8.0 # Mpc h^-1
def W(k,r):
    return (3.0*np.sin(k*R) - k*R*np.cos(k*R)) / (k*R)**3

sigma_8_sq_ln = romberg(P_log_space * k_log_space**3 * W(k_log_space,R)**2 / (2 * np.pi**2), dx=dlogk)
sigma_8_ln = np.sqrt(sigma_8_sq_ln)

#k_lin_space = np.linspace(0.001, 100., npoints)
#dk = k_lin_space[1] - k_lin_space[0]
#P_lin_space = 1.0 * k_lin_space**1.0
#sigma_8_sq_lin = romberg(P_lin_space * k_lin_space**2 * W(k_lin_space,R)**2 / (2 * np.pi**2), dx=dk)
#sigma_8_lin = np.sqrt(sigma_8_sq_lin)
#print("lin",sigma_8_lin)

print("log",sigma_8_ln)



