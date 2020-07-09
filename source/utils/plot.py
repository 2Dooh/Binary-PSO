import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def scatter_plot(ax_lim, ax, raw_data, title='', hold=True):
    x, y = raw_data[:, 0], raw_data[:, 1]
    ax.plot(x, y, "g.", label='genome')
    # ax.suptitle(title, size=15)
    # ax.title(title, loc='center')
    (xlim, ylim) = ax_lim
    ax.legend(loc=1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if not hold:
        plt.show()

def contour_plot(ax, coords, f_dict, title='', hold=True):
    (X, Y, Z) = coords
    global_minimums = f_dict['global minimum']
    global_maximums = f_dict['global maximum']
    if type(global_minimums) != type(None):
        global_minimums = np.reshape(global_minimums, (-1, 2))
        ax.plot(global_minimums[:, 0], global_minimums[:, 1], 'rx', label='global minimum', markersize=10)
    if type(global_maximums) != type(None):
        global_maximums = np.reshape(global_maximums, (-1, 2))
        ax.plot(global_maximums[:, 0], global_maximums[:, 1], 'bp', label='global maximum', markersize=10)

    ax.contour(X, Y, Z, 25, cmap=cm.seismic)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.set_title(f_dict['name'])

    # ax.legend(loc=1)
    if not hold:
        plt.show()

def scatter_3D(ax_lim, ax, X, Y, title='', hold=True):
    x = X[:, 0]
    y = X[:, 1]
    z = Y
    ax.scatter3D(x, y, z, label='Genome')
    (x_lim, y_lim, z_lim) = ax_lim
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    ax.legend()
    if not hold:
        plt.show()


def contour_3D(ax, coords, f_dict, title='', hold=True):
    (X, Y, Z) = coords

    ax.contour3D(X, Y, Z, 20, cmap=cm.rainbow)
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("F(x, y)")
    ax.set_title(f_dict['name'])

    # fig.colorbar(surf, shrink=0.5, aspect=10)
    # fig.suptitle(title)
        
    if not hold:
        plt.show()

def get_plot_data(f_dict):
    step = 1 if not f_dict['real valued'] else 0.25
    (lower_bound, upper_bound) = f_dict['D']
    x_axis = np.arange(lower_bound, upper_bound, step)
    y_axis = np.arange(lower_bound, upper_bound, step)
    n = len(x_axis)
    if n % 2 != 0:
        x_axis = x_axis[:-1]
        y_axis = y_axis[:-1]
    x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)

    if f_dict['multi dims']:
        # xy1, xy2 = x_mesh.copy(), y_mesh.copy()
        # num_half = len(xy1) // 2
        # xy1[:, :num_half], xy2[:, num_half:] = xy2[:, num_half:], xy1[:, :num_half].copy()

        # z1 = np.array(list(map(f_dict['function'], xy1)))[:, np.newaxis]
        # z2 = np.array(list(map(f_dict['function'], xy2)))[:, np.newaxis]
        # z_mesh = np.dstack((z1, z2))
        n = len(x_mesh)
        z_mesh = [f_dict['function'](np.array([x_mesh[i], y_mesh[i]])) for i in range(n)]
        z_mesh = np.array(z_mesh)
    else:
        z_mesh = np.array(f_dict['function']([x_mesh, y_mesh]))

    
    return (x_mesh, y_mesh, z_mesh)

# dict = f_dict.onemax_dict
# dict = f_dict.himmelblau_dict
# # dict = f_dict.cross_in_tray_dict
# dict = f_dict.rastrigin_dict
# coords = get_plot_data(dict)
# print("i'm here")
# # contour_plot(coords, dict)
# contour_3D(coords, dict, hold=False)