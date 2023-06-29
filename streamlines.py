import numpy as np
import matplotlib.pyplot as plt
from util import *

v = np.load("output/BaseLattice_last_u.npy")
obstacle = generate_obstacle_tensor("input/tortuosity/pdrop_08sig.png").cpu().numpy()
v[:, obstacle] = 0

d, nx, ny = v.shape
x = np.arange(nx)
y = np.arange(ny)
X, Y = np.meshgrid(x, y)
#
U = np.transpose(v[0, :, :])
V = np.transpose(v[1, :, :])
M = np.hypot(U, V)
magnitude = np.sqrt(U ** 2 + V ** 2)
n = 5
plt.imshow(obstacle.T, cmap='Greys')
plt.streamplot(X, Y, U, V, density=2, linewidth=.2, arrowstyle='-', color=magnitude, cmap=cmap)

plt.savefig('output/test.png', dpi=1500)
# plt.show()
