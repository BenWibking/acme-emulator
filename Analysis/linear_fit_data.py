import numpy as np
import numpy.linalg as linalg
from numba import jit
from math import exp
import time

@jit
def linear_kernel(X_i,X_j):
    return  X_i @ X_j.T

@jit
def compute_kernel(X):
    K = np.empty((X.shape[0],X.shape[0])) # kernel matrix
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            K[i,j] = linear_kernel(X[i,:],X[j,:])
    return K

@jit
def LOOE_norm(c,Ginv):
    """TODO: make this estimator aware of the noise properties of the data."""
    this_LOOE_norm = 0.
    for i in range(c.shape[0]):
        this_LOOE_norm += (c[i,0] / Ginv[i,i])**2
    return this_LOOE_norm

def fit_data_with_kernel(Y,K):
    """fit data with a least-squares model with kernel K, observations Y."""
    eigval, Q = np.linalg.eigh(K)

    # define figure of merit
    def this_problem_err(regularizer):
        Ginv = Q @ np.diag(1.0/(eigval + regularizer)) @ Q.T
        c = Ginv @ Y
        return LOOE_norm(c,Ginv)

    # search for minimum, varying regularization parameter
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(this_problem_err)
    best_regularizer = result.x
    
    # return best solution
    Ginv = Q @ np.diag(1.0/(eigval + best_regularizer)) @ Q.T
    c = Ginv @ Y
    rms_err_norm = np.sqrt(LOOE_norm(c,Ginv)/c.shape[0])

    return c,rms_err_norm,best_regularizer

@jit
def model_data_at_point(X_input, X_model_i, c):
    """return model prediction at point X_model_i given c trained on X_input"""
    f = 0.
    for i in range(c.shape[0]):
        f += linear_kernel(X_input[i,:], X_model_i) * c[i]
    return f

@jit
def model_data(X_input, X_model, c):
    Y_model = np.zeros(X_model.shape[0])
    for i in range(X_model.shape[0]):
        for j in range(c.shape[0]):
            Y_model[i] += linear_kernel(X_input[j,:], X_model[i,:]) * c[j]
    return Y_model

def compute_labels(x,x0=0.0):
    dx = x - x0
    return np.c_[dx**0,dx**1,dx**2,dx**3,dx**4,dx**5,dx**6]
#    return np.c_[dx**0,dx**1,dx**2,dx**3,dx**4,dx**5]
#    return np.c_[dx**0,dx**1,dx**2,dx**3,dx**4]
#    return np.c_[dx**0,dx**1,dx**2,dx**3]
#    return np.c_[dx**0,dx**1,dx**2]
#    return np.c_[dx**0,dx**1]
#    return np.c_[dx]

def fit_data(input_file):
    """fit data with a linear least-squares model"""
    x,y = np.loadtxt(input_file,unpack=True)

    # solve: min_c { |Y - Kc|^2 + Lambda |c|^2 }
    # solution: c = K^{-1} Y
    Y = np.matrix(y).T # column vector
    X = compute_labels(x)
    K = compute_kernel(X)
    c, rms_LOOE_norm, best_lambda = fit_data_with_kernel(Y,K)

    print('rms leave-one-out error: {}'.format(rms_LOOE_norm))
    print('best L2-regularization hyperparameter = {}'.format(best_lambda))

    ## only if linear kernel:
    w = X.T @ c # components of w are the linear coefficients, i.e. Y = Xw
    print('linear coefficients w = {}'.format(w))

    ## plot model
    x_model = np.linspace(x.min(),x.max(),50)
    X_model = compute_labels(x_model)
    print('computing model predictions...',end='',flush=True)
    y_model = model_data(X, X_model, c)
    print('done.')

    y_true = lambda x: 2.0*x + 0.2*np.sin(3.0*np.pi*x)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(x,y,label='data',s=0.5)
    plt.plot(x_model,y_model,label='model')
    plt.plot(x_model,y_true(x_model),label='truth')
#    plt.scatter(x,y/y_true(x) - 1.0,label='data residuals',s=0.5)
#    plt.plot(x_model,y_model/y_true(x_model) - 1.0,label='model error')
    plt.ylabel('$f(x)$')
    plt.xlabel('x')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('model_fit.pdf')

    plt.figure()
    plt.scatter(x,y/y_true(x) - 1.0,label='data residuals',s=0.5)
    plt.plot(x_model,y_model/y_true(x_model) - 1.0,label='model error')
    plt.ylabel('fractional error in $f(x)$')
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

    
