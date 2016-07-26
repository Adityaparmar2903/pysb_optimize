from pysb.examples.earm_1_3 import model
import numpy as np
from pysb.integrate import Solver
import scipy.optimize
import numdifftools as nd
import pickle
import os
import sys


num_timepoints = 101
num_dim = 3
t = np.linspace(0, 21600, num_timepoints)
method_list = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B',
               'TNC', 'COBYLA', 'SLSQP', 'differential evolution']

sol = Solver(model, t, use_analytic_jacobian=True, nsteps = 10000)

sol.run()

data = np.zeros((num_timepoints, 1))

obs = model.observables['CPARP_total']
obs_max = np.max(sol.yobs[obs.name])
rand_norm = np.random.randn(num_timepoints)
noise = rand_norm * 50000
noisy_obs = noise + sol.yobs[obs.name]
norm_noisy_data = noisy_obs / obs_max
data = np.load('data_earm.npy')

p_to_fit = [p for p in model.parameters
                     if p.name[0] == 'k' and p.name[1] != 'd' and p.name[1] != 's']

num_obj_calls = 0

def obj_func(x):
    global num_obj_calls
    num_obj_calls += 1
    lin_x = 10 ** x
    p = x - x_test
    for p_ix, pp in enumerate(p_to_fit):
        pp.value = lin_x[p_ix]
    sol.run()
    total_err = 0
    observed = [model.observables['CPARP_total']]
    for obs_ix, obs in enumerate(observed):
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

bds = []

nominal_values = np.array([p.value for p in p_to_fit])
x_test = np.log10(nominal_values)
for ind in range(len(p_to_fit)):
    low = x_test[ind] - 2
    up = x_test[ind] + 2
    bd = (low, up)
    bds.append(bd)


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
    ini_val = np.load('initial_values_earm.npy')
    if method_id == 11:
        for i in range(from_idx, to_idx):
            result = scipy.optimize.differential_evolution(obj_func, bds)
            fname = '%s_%d.pkl' % (method_id,i)
            with open(fname, 'wb') as fh:
                pickle.dump(result, fh)
            func_eval = num_obj_calls
            global num_obj_calls
            num_obj_calls = 0
            fname = 'eval-%s_%d.pkl' % (method_id, i)
            with open(fname, 'wb') as fh:
                pickle.dump(func_eval, fh)
    else:
        for i in range(from_idx, to_idx):
            result = scipy.optimize.minimize(obj_func, ini_val[i], method=meth)
            fname = '%s_%d.pkl' % (method_id,i)
            with open(fname, 'wb') as fh:
                pickle.dump(result, fh)
            func_eval = num_obj_calls
            global num_obj_calls
            num_obj_calls = 0
            fname = 'eval-%s_%d.pkl' % (method_id, i)
            with open(fname, 'wb') as fh:
                pickle.dump(func_eval, fh)
