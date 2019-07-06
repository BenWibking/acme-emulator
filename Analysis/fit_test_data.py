import numpy as np
import numpy.linalg as linalg
import sys
from numba import jit
from math import exp
import time

@jit
def linear_kernel(X_i,X_j):
    return  X_i @ X_j.T

@jit
def polynomial_kernel(X_i,X_j,h):
    return (1.0 + (X_i @ X_j.T))**6

@jit
def exponential_kernel(X_i,X_j,h):
    """compute exponential kernel matrix: K_ij = exp(-|X_i - X_j|^2)"""
    w = (X_i - X_j)
    # TODO: write 'np.diag(h)' in factor analysis form (allow parameter space rotations)
    return exp(-(w @ (np.diag(h) @ w.T))) # can this be factored for efficiency?

@jit
def taylor_expanded_exponential_kernel(X_i,X_j,h):
    """compute approximation to exponential kernel matrix: K_ij = exp(-|X_i - X_j|^2)"""
    w = (X_i - X_j)
    t = (1.0 + (w @ (np.diag(h) @ w.T)))
    return 1.0/(t*t*t*t)

@jit
def generic_kernel(X_i,X_j,h):
    """[choose kernel here.]"""
#    return taylor_expanded_exponential_kernel(X_i,X_j,h)
    return exponential_kernel(X_i,X_j,h)
#    return polynomial_kernel(X_i,X_j,h)
#    return linear_kernel(X_i,X_j)

@jit
def compute_kernel(X,h):
    K = np.empty((X.shape[0],X.shape[0])) # kernel matrix
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            K[i,j] = generic_kernel(X[i,:],X[j,:],h)
    return K

@jit
def LOOE_norm(c,Ginv):
    """TODO: make this estimator aware of the noise properties of the data."""
    this_LOOE_norm = 0.
    for i in range(c.shape[0]):
        this_LOOE_norm += (c[i,0] / Ginv[i,i])**2
    return this_LOOE_norm

def fit_data_with_kernel(Y,K,Lambda):
    """fit data with a least-squares model with kernel K, observations Y.
    If noise of observations is Gaussian, Lambda = 2*var(x)."""
    eigval, Q = np.linalg.eigh(K)
    D = np.zeros(Q.shape)
    for i in range(D.shape[0]):
        D[i,i] = 1.0/(eigval[i] + Lambda)
    Ginv = Q @ D @ Q.T
    c = Ginv @ Y
    rms_err_norm = np.sqrt(LOOE_norm(c,Ginv)/c.shape[0]) # this is wrong if heteroscedastic!
    
    return c,rms_err_norm

def fit_hyperparameters(Y,X):
    N=Y.shape[0]
    nparams=X.shape[1]+1
    print('number of observations: {}'.format(N))
    print('number of hyperparameters: {}'.format(nparams))
    scale_guess = [1.0/(np.max(X[:,i]) - np.min(X[:,i])) for i in range(nparams-1)]
    lambda_guess = (2.0*np.var(Y))

    def outer_opt(h):
        K = compute_kernel(X,h[1:])
        c, rms_LOOE_norm = fit_data_with_kernel(Y,K,h[0])
        return rms_LOOE_norm

    from scipy.optimize import minimize
    print('optimizing hyperparameters...',end='',flush=True)
    start_time = time.time()
#    x0 = np.ones(nparams)
    x0 = np.random.uniform(size=nparams, high=np.concatenate([[lambda_guess], scale_guess]))
    bounds = [(0.,None) for i in range(nparams)] # all scale parameters should be positive
    res =  minimize(outer_opt, x0, method='L-BFGS-B',
                    bounds = bounds,
                    options={'gtol':1e-3, 'eps':1e-5, 'maxiter':200})
    best_h = (res.x[1:])
    best_lambda = (res.x[0])
    print('done in {} seconds.'.format(time.time() - start_time))

    K = compute_kernel(X,best_h)
    c, rms_LOOE_norm = fit_data_with_kernel(Y,K,best_lambda)

    return c, rms_LOOE_norm, best_lambda, best_h

@jit
def model_data(X_input, X_model, c, h):
    Y_model = np.zeros(X_model.shape[0])
    for i in range(X_model.shape[0]):
        for j in range(c.shape[0]):
            Y_model[i] += generic_kernel(X_input[j,:], X_model[i,:], h) * c[j]
    return Y_model

def compute_labels(x,x0=0.0):
    dx = x - x0
    return dx

def fit_data(input_file):
    """fit data with a linear least-squares model"""
    data = np.loadtxt(input_file)
    x = data[:,:-2] # can be a matrix
    y = data[:, -2]
    y_true = data[:,-1]

    # solve: min_c { |Y - Kc|^2 + Lambda |c|^2 }
    # solution: c = K^{-1} Y
    Y = np.matrix(y).T # column vector
    X = compute_labels(x)

    # fit hyperparameters with random restarts
    rms_LOOE_norm = np.inf
    num_restarts = 10
    for i in range(num_restarts):
        print('restart #{}...'.format(i))
        this_c, this_rms_LOOE_norm, this_lambda, this_h = fit_hyperparameters(Y,X)
        print("rms_LOOE: {}, lambda^2 = {}, h = {}".format(this_rms_LOOE_norm, this_lambda, this_h))
        if this_rms_LOOE_norm < rms_LOOE_norm:
            rms_LOOE_norm = this_rms_LOOE_norm
            c = this_c
            best_lambda = this_lambda
            best_h = this_h
        print('')

    print('rms leave-one-out error: {}'.format(rms_LOOE_norm))
    print('best L2-regularization hyperparameter = {}'.format(best_lambda))
    print('best scale hyperparameter = {}'.format(best_h))

    ## plot model
    print('computing model predictions...',end='',flush=True)
    y_model_x = model_data(X, X, c, best_h)
    print('done.')

    print('rms model residuals = {}'.format(np.sqrt(np.mean((y_model_x - y_true)**2))))
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(x[:,0],y,label='data')
    plt.scatter(x[:,0],y_model_x,label='model',marker='+')
    plt.ylabel('$f(x)$')
    plt.xlabel('x[0]')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('model_fit.pdf')

    plt.figure()
    plt.scatter(x[:,0], y_model_x - y_true,label='model residuals')
    plt.ylabel('error in $f(x)$')
    plt.xlabel('x')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('model_fit_residuals.pdf')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    args = parser.parse_args()
    fit_data(args.input_file)

    
