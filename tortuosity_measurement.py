import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import norm
from numpy import gradient, sqrt
from scipy import integrate
from scipy.interpolate import RegularGridInterpolator
from util import *

v = np.load("output/BaseLattice_last_u.npy")
obstacle = generate_obstacle_tensor("input/halfcell.png").cpu().numpy()
v[:, obstacle] = 0
U = v[0]
V = v[1]
x = np.arange(v.shape[1])
y = np.arange(v.shape[2])

# Interpolation of vector field
U_interp = RegularGridInterpolator((x, y), U)
V_interp = RegularGridInterpolator((x, y), V)


def func(point, t):
    return [U_interp(point)[0], V_interp(point)[0]]


# plotting
fig, ax = plt.subplots()
ax.streamplot(x, y, U.T, V.T, broken_streamlines=False)


for i in range(0, v.shape[2], 10):
    print(i * 100 / v.shape[2])
    # initial seed
    y0 = [v.shape[1] - 1, i] # Start from last element on the right (outflow) for each y

    # t parameter
    path = [y0]

    while True:
        t = np.array([0.1, 0])
        step = integrate.odeint(func, path[-1], t)
        path.append(step[-1])
        if np.abs(path[-1][0] - 0) < 1 or np.abs(path[-1][1] - 0) < 1:  # if reach boundary
            break
        if norm(path[-1] - path[-2]) < 1e-4 or len(path) > 10000:  # If reached end of streamline (v = 0)
            break

        ax.plot(np.asarray(path)[:, 0], np.asarray(path)[:, 1], 'r--')


plt.show()