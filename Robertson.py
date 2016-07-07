from pysb.examples.robertson import model
import numpy as np
from matplotlib import pyplot as plt
from pysb.integrate import Solver
import scipy.optimize
from pyDOE import *
import os
import numdifftools as nd
from pysb.sensitivity import Sensitivity

colors = ['red', 'darkgreen', 'blue', 'cyan', 'magenta',
         'yellow', 'orange', 'sienna', 'black', 'lawngreen', 'slategrey']
method_list = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B',
               'TNC', 'COBYLA', 'SLSQP', 'Newton-CG', 'trust-ncg', 'dogleg']

num_ini = 5
num_timepoints = 101
num_dim = 3
t = np.linspace(0, 200, num_timepoints)
num_obj_calls = 0

data = np.zeros((num_timepoints, len(model.observables)))
obj_func_val = np.ones((len(method_list), num_ini))
obj_func_val[:] = np.nan
func_eval = np.ones((len(method_list), num_ini))
func_eval[:] = np.nan
vector = np.ones((len(method_list), num_ini, 3))
ind = np.arange(num_ini)

p_to_fit = [p for p in model.parameters if p.name[0] == 'k']
p_to_fit_indices = [model.parameters.index(p) for p in p_to_fit]
nominal_values = np.array([p.value for p in p_to_fit])
x_test = np.log10(nominal_values)

sol = Solver(model, t, use_analytic_jacobian=True, nsteps=10000)
sol.run()
plt.ion()
#sens = Sensitivity(model, t)

def plot_simulate(model, t):
    for obs_ix, obs in enumerate(model.observables):
        obs_max = np.max(sol.yobs[obs.name])
        plt.plot(t, sol.yobs[obs.name] / obs_max, label=obs.name,
             color=colors[obs_ix])

def gen_synth_data(model, t):
    for obs_ix, obs in enumerate(model.observables):
        rand_norm = np.random.randn(len(t))
        sigma = 0.1
        obs_max = np.max(sol.yobs[obs.name])
        noise = rand_norm * sigma * sol.yobs[obs.name]
        noisy_obs = noise + sol.yobs[obs.name]
        norm_noisy_data = noisy_obs / obs_max
        plt.plot(t, norm_noisy_data, linestyle='', marker='.',
                color=colors[obs_ix])
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
        total_err += np.sum(p[np.where(p > 3)] - 3)*1000
        total_err += np.sum(-3 - p[np.where(p < -3)])*1000
    except Exception as e:
        print "couldn't apply constraints"
    print total_err
    return total_err

def generate_init(ns):
    ini_val = lhs(len(p_to_fit), ns)
    means = x_test
    stdvs = 2*np.ones(len(p_to_fit))
    ini_val = means + 2*stdvs*(ini_val-0.5)
    return ini_val

ini_val = generate_init(num_ini)

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

num_jac_calls = 0
def jac_func(x):
    global num_jac_calls
    num_jac_calls += 1
    lin_x = 10 ** x
    # Initialize the model to have the values in the parameter array
    for p_ix, p in enumerate(p_to_fit):
        p.value = lin_x[p_ix]
    sens.run()
    ysens_view = sens.ysens[:, :, p_to_fit_indices]
    dgdp = np.zeros(len(p_to_fit))
    for y_ix in range(sens.yodes.shape[1]):
        dgdy = np.dot(2 * (sens.yodes[:, y_ix] - data[:, y_ix]),
                      ysens_view[:, y_ix, :])
        dgdp += dgdy
    return dgdp

def hess_func(x):
    jaco = nd.Jacobian(jac_func)(x)
    return jaco

def fit():
    for j in range(num_ini):
        x0 = ini_val[j]
        for k, meth in enumerate(method_list):
            result = scipy.optimize.minimize(obj_func, x0, method=meth,
                     jac=Jacob, hess = Hessi)
            obj_func_val[k][j] = result.fun
            func_eval[k][j] = num_obj_calls
            vector[k][j] = result.x
            global num_obj_calls
            num_obj_calls = 0

def fit_sense():
    for j in range(num_ini):
        x0 = ini_val[j]
        for k, meth in enumerate(method_list):
            result = scipy.optimize.minimize(obj_func, x0, method=meth,
                     jac=jac_func, hess = Hessi)
            obj_func_val[k][j] = result.fun
            func_eval[k][j] = num_obj_calls
            vector[k][j] = result.x
            global num_obj_calls
            num_obj_calls = 0



plot_simulate(model, t)
gen_synth_data(model, t)
fit()
print "True values (in log10 space):", x_test
print "Nominal error:", obj_func(x_test)

def plot_obj_func():
    plt.figure()
    for i, meth in enumerate(method_list):
        plt.plot(ind+1, sorted(obj_func_val[i]), linestyle = 'solid',
                 marker='.', markeredgewidth=0.0, color = colors[i], label = meth)
        plt.legend()
        plt.xlabel("Run Index")
        plt.ylabel("Objective Function Value (log)")
        plt.grid(True)

def plot_obj_func_sort(index):
    plt.figure
    inds = obj_func_val[index].argsort()
    for j in range(len(method_list)):
        obj_func_val[j] = obj_func_val[j][inds]
    for i, meth in enumerate(method_list):
        plt.plot(ind+1, sorted(obj_func_val[i]), linestyle = 'solid',
                 marker='.', markeredgewidth=0.0, color = colors[i] , label = meth)
        plt.legend()
        plt.xlabel("Run Index")
        plt.ylabel("Objective Function Value (log)")
        plt.grid(True)

plot_obj_func()
plot_obj_func_sort()



"""
plt.figure()
# Plot the original data
plt.plot(t, data, linestyle='', marker='.', color='k')
# Plot BEFORE
# Set parameter values to start position
for p_ix, p in enumerate(p_to_fit):
    p.value = 10 ** x0[p_ix]
sol.run()
plt.plot(t, sol.y, color='r')
# Plot AFTER
# Set parameter values to final position
for p_ix, p in enumerate(p_to_fit):
    p.value = 10 ** result.x[p_ix]
sol.run()
plt.plot(t, sol.y, color='g')
"""
