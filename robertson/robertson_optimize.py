from pysb.examples.robertson import model
import numpy as np
from pysb.integrate import Solver
import scipy.optimize
from pyDOE import *
import os
import sys
import numdifftools as nd
from matplotlib import pyplot as plt
import pickle
from pysb.sensitivity import Sensitivity

method_list = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B',
               'TNC', 'COBYLA','SLSQP', 'Newton-CG', 'trust-ncg', 'dogleg', 'differential evolution']

num_timepoints = 101
num_dim = 3
t = np.linspace(0, 200, num_timepoints)
num_obj_calls = 0

data = np.zeros((num_timepoints, len(model.observables)))

p_to_fit = [p for p in model.parameters if p.name[0] == 'k']
p_to_fit_indices = [model.parameters.index(p) for p in p_to_fit]
nominal_values = np.array([p.value for p in p_to_fit])
x_test = np.log10(nominal_values)

sol = Solver(model, t, use_analytic_jacobian=True, nsteps=10000)
sol.run()
#sens = Sensitivity(model, t)
plt.ion()

def gen_synth_data(model, t):
    for obs_ix, obs in enumerate(model.observables):
        rand_norm = np.random.randn(len(t))
        sigma = 0.1
        obs_max = np.max(sol.yobs[obs.name])
        noise = rand_norm * sigma * sol.yobs[obs.name]
        noisy_obs = noise + sol.yobs[obs.name]
        norm_noisy_data = noisy_obs / obs_max
        data[:, obs_ix] = noisy_obs

def obj_func(x):
    global num_obj_calls
    num_obj_calls += 1
    p = x - x_test
    lin_x = 10 ** x
    for p_ix, pp in enumerate(p_to_fit):
        pp.value = lin_x[p_ix]
    sol.run()
    total_err = 0
    for obs_ix, obs in enumerate(model.observables):
        y = sol.yobs[obs.name]
        total_err += np.sum((y - data[:, obs_ix])**2)
    try:
        total_err += np.sum(p[np.where(p > 2)] - 2)*1000
        total_err += np.sum(-2 - p[np.where(p < -2)])*1000
    except Exception as e:
        print "couldn't apply constraints"
    print total_err
    return total_err

def Jacob(x):
    if np.any(np.isnan(x)):
        jaco = np.zeros(x.shape)
        jaco[:] = np.nan
        return jaco
    jaco = nd.Jacobian(obj_func)(x)
    return jaco[0]

def Hessi(x):
    if np.any(np.isnan(x)):
        hes = np.zeros((len(x), len(x)))
        hes[:] = np.nan
        return hes
    hes = nd.Hessian(obj_func)(x)
    return hes

def jac_func(x):
    #global num_jac_calls
    #num_jac_calls += 1
    lin_x = 10 ** x
    # Initialize the model to have the values in the parameter array
    for p_ix, p in enumerate(p_to_fit):
        p.value = lin_x[p_ix]
    sens.run()
    dgdp = np.zeros(len(model.parameters))

    for obs_ix, obs in enumerate(model.observables):
        yobs = sens.yobs[obs.name]
        ysens = sens.yobs_sens[obs.name]
        y = data[:, obs_ix]
        dgdy = np.tensordot(ysens, 2 * (yobs -  y), axes=(0, 0))
        dgdp += dgdy
    dgdp[0:3] = dgdp[0:3] * np.log(10) * lin_x
    return dgdp[0:3]

def hess_func(x):
    jaco = nd.Jacobian(jac_func)(x)
    return jaco

#gen_synth_data(model, t)
data = np.load('data.npy')
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print 'Not enough input arguments.'
        sys.exit()
    from_idx = int(sys.argv[1])
    to_idx = int(sys.argv[2])
    if to_idx < from_idx:
        print 'Invalid from-to pair.'
        sys.exit()
    method_id = int(sys.argv[3])
    if method_id >= len(method_list):
        print 'Invalid method id.'
        sys.exit()
    meth = method_list[method_id]
    ini_val = np.load('initial_values.npy')
    if method_id == 11:
        for i in range(from_idx, to_idx):
            result = scipy.optimize.differential_evolution(obj_func, [(-3.39, 0.61),(5.47, 9.47),(2,6)])
            fname = 'Rob-%s_%d.pkl' % (method_id,i)
            with open(fname, 'wb') as fh:
                pickle.dump(result, fh)
            func_eval = num_obj_calls
            global num_obj_calls
            num_obj_calls = 0
            fname = 'Rob-eval-%s_%d.pkl' % (method_id, i)
            with open(fname, 'wb') as fh:
                pickle.dump(func_eval, fh)
    else:
        for i in range(from_idx, to_idx):
	    result = scipy.optimize.minimize(obj_func, ini_val[i], method=meth, jac=Jacob, hess=Hessi)
            fname = 'Rob-%s_%d.pkl' % (method_id,i)
            with open(fname, 'wb') as fh:
                pickle.dump(result, fh)
            func_eval = num_obj_calls
            global num_obj_calls
            num_obj_calls = 0
            fname = 'Rob-eval-%s_%d.pkl' % (method_id, i)
            with open(fname, 'wb') as fh:
                pickle.dump(func_eval, fh)
