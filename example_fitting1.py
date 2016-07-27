from pysb.examples.robertson import model
import numpy as np
from matplotlib import pyplot as plt
from pysb.integrate import Solver
import scipy.optimize
import numdifftools as nd
import pickle

num_timepoints = 101
num_dim = 3
t = np.linspace(0, 200, num_timepoints)
norm = np.zeros(len(model.observables))
sol = Solver(model, t, use_analytic_jacobian=True)

sol.run()
plt.ion()

colors2 = ['blue', 'cornflowerblue', 'cyan', 'darkgreen', 'lawngreen', 'darkorange',
          'yellow', 'plum', 'red', 'magenta', 'sienna', 'black']
colors = ['red', 'green', 'blue']
colors1 = ['pink', 'lightgreen', 'skyblue']
data = np.zeros((num_timepoints, len(model.observables)))

plt.figure()
for obs_ix, obs in enumerate(model.observables):
    obs_max = np.max(sol.yobs[obs.name])
    plt.plot(t, sol.yobs[obs.name] / obs_max, label=obs.name,
             color=colors[obs_ix])
    plt.legend()
plt.suptitle('Simulation of the model equations')
plt.xlabel('Time (in seconds)')
plt.ylabel('Normalized Concentration')

plt.figure()
for obs_ix, obs in enumerate(model.observables):
    obs_max = np.max(sol.yobs[obs.name])
    plt.plot(t, sol.yobs[obs.name] / obs_max, label=obs.name,
             color=colors[obs_ix])
    norm_noisy_data = np.load('data_plot_robertson-%d.npy' %(obs_ix))
    plt.plot(t, norm_noisy_data, linestyle='', marker='.',
             color=colors[obs_ix])
    plt.legend()
plt.suptitle('Synthetic noisy data')
plt.xlabel('Time (in seconds)')
plt.ylabel('Normalized Concentration')

data = np.load('data_robertson.npy')

p_to_fit = [p for p in model.parameters
                     if p.name in ['k1', 'k2', 'k3']]

def obj_func(x):
    lin_x = 10 ** x
    print x
    for p_ix, p in enumerate(p_to_fit):
        p.value = lin_x[p_ix]
    sol.run()
    total_err = 0
    for obs_ix, obs in enumerate(model.observables):
        y = sol.yobs[obs.name]
        total_err += np.sum((y - data[:, obs_ix])**2)
    print total_err
    return total_err


def Jacob(x):
    if np.any(np.isnan(x)):
        jaco = np.zeros(x.shape)
        jaco[:] = np.nan
        return jaco
    jaco = nd.Jacobian(obj_func)(x)
    return jaco[0]

nominal_values = np.array([p.value for p in p_to_fit])
x_test = np.log10(nominal_values)
print "True values (in log10 space):", x_test
print "Nominal error:", obj_func(x_test)

ini_val = np.load('initial_values.npy')
x0 = ini_val[0]
result = scipy.optimize.minimize(obj_func, x0, method='Nelder-Mead', options = {'disp' : True})


plt.figure()
plt.plot(t, data, linestyle='', marker='.', color='k')
plt.xlabel('Time (in seconds)')
plt.ylabel('Concentration')
plt.suptitle('Data to fit')

plt.figure()
plt.plot(t, data, linestyle='', marker='.', color='k')
for p_ix, p in enumerate(p_to_fit):
    p.value = 10 ** x0[p_ix]
sol.run()
for obs_ix, obs in enumerate(model.observables):
    plt.plot(t, sol.yobs[obs.name], color=colors1[obs_ix],
    label = obs.name)
plt.legend()
plt.xlabel('Time (in seconds)')
plt.ylabel('Concentration')
plt.suptitle('Initial guess')

plt.figure()
plt.plot(t, data, linestyle='', marker='.', color='k')
for p_ix, p in enumerate(p_to_fit):
    p.value = 10 ** x0[p_ix]
sol.run()
for obs_ix, obs in enumerate(model.observables):
    plt.plot(t, sol.yobs[obs.name], color=colors1[obs_ix])
for p_ix, p in enumerate(p_to_fit):
    p.value = 10 ** result.x[p_ix]
sol.run()
for obs_ix, obs in enumerate(model.observables):
    plt.plot(t, sol.yobs[obs.name], color=colors[obs_ix],
    label = obs.name)
plt.ylim(-0.2,1)
plt.legend()
plt.xlabel('Time (in seconds)')
plt.ylabel('Concentration')
plt.suptitle('Final fitting')
