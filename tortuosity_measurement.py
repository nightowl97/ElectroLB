import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import norm
from numpy import gradient, sqrt
from scipy import integrate
from scipy.interpolate import RegularGridInterpolator
from util import *

v = np.load("output/BaseLattice_last_u.npy")
obstacle = generate_obstacle_tensor("input/pdrop/pdroplarge_3.png").cpu().numpy()
v[:, obstacle] = 0
U = v[0]
V = v[1]
x = np.arange(v.shape[1])
y = np.arange(v.shape[2])

# Interpolation of vector field
U_interp = RegularGridInterpolator((x, y), U)
V_interp = RegularGridInterpolator((x, y), V)


def func(point):
    return [U_interp(point)[0], V_interp(point)[0]]


# plotting
fig, ax = plt.subplots()
# ax.streamplot(x, y, U.T, V.T)

paths = []
for i in range(0, v.shape[2], 5):
    # initial seed
    y0 = np.asarray([v.shape[1] - 1, i])  # Start from last element on the right (outflow) for each y

    # initialize path
    path = [y0]

    while True:
        t = -100  # step backwards
        # Calculate next step from previous step using euler method
        next_step = path[-1] + np.asarray(func(path[-1])) * t
        path.append(next_step)
        if path[-1][0] < 20:  # TODO: Consider this when calculating reference length
            paths.append(path)
            break
        if np.abs(path[-1][1] - 0) < 10 or np.abs(path[-1][1] - v.shape[2]) < 10:  # if reached top boundary
            break
        if norm(path[-1] - path[-2]) < 1e-6 or len(path) > 100000:  # If reached end of streamline (v = 0)
            break
        print(f"Progress: {int(i * 100 / v.shape[2])}%, \tCurrent x: {path[-1][0]}\tCurrent y: {path[-1][1]}", sep='', end="\r", flush=True)

# Calculate lengths of streamlines
print("Calculating lengths...")
lengths = []
for path in paths:
    path_array = np.asarray(path)
    dxdt = np.gradient(path_array[:, 0])
    dydt = np.gradient(path_array[:, 1])
    t_path = np.linspace(0, v.shape[1], len(path))
    length = np.sum(np.sqrt(dxdt ** 2 + dydt ** 2))
    lengths.append(length)

for path in paths:
    ax.plot(np.asarray(path)[:, 0], np.asarray(path)[:, 1], 'r--', linewidth=0.2)

plt.imshow(np.invert(obstacle.T), cmap='gray')
plt.savefig("output/Streamlines.png", dpi=1200)

lengths = np.asarray(lengths)
tortuosity = np.mean(lengths) / v.shape[1]

print(f"Measured Tortuosity: {tortuosity}")
