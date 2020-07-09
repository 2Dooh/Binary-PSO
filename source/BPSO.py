import numpy as np

from optimizers.GA import *
from utils.plot import *

np.set_printoptions(suppress=True)  # Prevent numpy exponential notation on print, default False

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


from numpy.random import rand
def compute_velocity(v, g, P, p, params):
    (c1, c2) = params['ac']
    w = params['iw']
    r_p, r_g = rand(), rand()
    new_v = w*v + c1*r_p * (p - P) + c2*r_g * (g - P)
    return new_v


def selection(f_current, f_prev, maximize=False):
    if maximize:
        return np.where(f_current > f_prev)
    return np.where(f_current < f_prev)

def ring_selection(f_p, maximize=False):
    comparer = max if maximize else min
    indices = []
    n = len(f_p)
    for i in range(n - 1):
        neighbors = (f_p[i-1], f_p[i], f_p[i+1])
        idx = neighbors.index(comparer(neighbors))
        indices.append(idx + (i-1))

    neighbors_last = (f_p[n-2], f_p[n-1], f_p[0])
    idx_last = neighbors_last.index(comparer(neighbors_last))
    if idx_last != 2:
        indices.append(idx_last + n-2)
    else:
        indices.append(0)

    return indices

def star_selection(f_p, maximize=False):
    comparer = np.argmax if maximize else np.argmin
    return comparer(f_p, axis=0)

def select_best_neighbors(mode, f_p, maximize):
    assert(mode != 'star' or mode != 'ring')

    if mode == 'star':
        return star_selection(f_p, maximize)
    if mode == 'ring':
        return ring_selection(f_p, maximize)
    

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import uniform
def optimize(params, plot=0, print_scr=False):
    """

    """

    # Initialize required parameters from dictionary
    num_inds = params['N']
    max_gen = params['g']
    seed = params['s']
    maximize = params['maximize']

    selection_mode = params['sm']

    f_dict = params['f']    # Dictionary of fitness function data
    num_params = f_dict['d']
    (lower_bound, upper_bound) = f_dict['D']
    real_valued = f_dict['real valued']

    # Plot search space
    plottable = plot and num_params == 2
    if plottable:
        data = get_plot_data(f_dict)
        fig, ax = plt.subplots()
        if plottable and plot == 2:
            ax = Axes3D(fig)
    
    # Initialize
    comparer = np.argmax if maximize else np.argmin
    np.random.seed(seed)
    epsilon = 10**-5
    P = initialize(num_inds, num_params, 
                   domain=(lower_bound, upper_bound), 
                   real_valued=real_valued)
    f_P = None
    p = P.copy()
    v = uniform(low=-abs(upper_bound-lower_bound), 
                high=abs(upper_bound-lower_bound), 
                size=(num_inds, num_params)).astype(np.float64)
    gen = 0
    num_f_func_calls = 0
    #
    while not pop_converge(P):
        gen += 1
        if max_gen_reached(gen, max_gen):
            break

        # Evaluate
        f_P = evaluate(P, f_dict['function'])
        f_p = evaluate(p, f_dict['function'])

        num_f_func_calls += len(f_P) * 2

        # Selection
        selected_indices = selection(f_P, f_p, maximize)
        p[selected_indices] = P[selected_indices]
        f_p[selected_indices] = f_P[selected_indices]

        g_indices = select_best_neighbors(selection_mode, f_p, maximize)
        g = p[g_indices]

        # Variate
        v = compute_velocity(v, g, P, p, params)
        P = (sigmoid(v) > 0.5).astype(np.int)
        #

        # Visualize / log result
        if print_scr and gen % 100 == 0:
            print('## Gen {}: {} (Fitness: {})'.format(gen, P[comparer(f_P)].reshape(1, -1), 
                                                       f_P[comparer(f_P)]))
        if plottable:
            ax.clear()
            if plot == 1:
                contour_plot(ax, data, f_dict, hold=True)
                ax_lim = (xlim, ylim) = ax.get_xlim(), ax.get_ylim()
                scatter_plot(ax_lim, ax, P, hold=True)
            else:
                contour_3D(ax, data, f_dict, hold=True)
                ax_lim = (xlim, ylim, zlim) = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
                scatter_3D(ax_lim, ax, P, f_P, hold=True)
            plt.pause(epsilon)
        #

    if plottable: 
        plt.show()     

    solution =  P[comparer(f_P)].reshape(1, -1).flatten()
    opt_sol_found = None

    optimize_goal = 'global maximum' if maximize else 'global minimum'
    if type(f_dict[optimize_goal]) != type(None):
        diffs = np.abs(f_dict[optimize_goal] - solution).sum(axis=1)
        opt_sol_found = len(np.where(diffs <= num_params*epsilon)[0]) != 0

    result = { 'solution' : solution, 
               'evaluate function calls' : num_f_func_calls, 
               'global optima found' : opt_sol_found }
    return result


def get_parameters(**params):
    mode = 'star' if params['mode'] == "" or params['mode'] not in ['star', 'ring'] else params['mode']
    default_params = { 'N' : params['N'],
                       's' : params['s'],
                       'iw' : 0.7298,
                       'ac' : (1.49618, 1.49618),
                       'g' : params['g'],
                       'sm' : mode,
                       'maximize' : params['maximize'],
                       'f' : params['f']}
    return default_params


