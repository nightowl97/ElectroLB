import numpy as np
import matplotlib.pyplot as plt
from util import *

v = np.load("input/mmrfb_u.npy")
obstacle = generate_obstacle_tensor("input/mmrfb_original.png").cpu().numpy()
v[:, obstacle] = 0

d, nx, ny = v.shape
x = np.linspace(0, 1000, nx)
y = np.linspace(0, 872, ny)
X, Y = np.meshgrid(x, y)
#
U = np.transpose(v[0, :, :])
V = np.transpose(v[1, :, :])
M = np.hypot(U, V)
magnitude = np.sqrt(U ** 2 + V ** 2)
n = 5
plt.imshow(obstacle.T == True, cmap='Greys')
plt.streamplot(X, Y, U, V, density=20, linewidth=.1, arrowstyle='-', color=magnitude, cmap=cmap, broken_streamlines=False)

plt.savefig('output/test_3.png', dpi=1500)
