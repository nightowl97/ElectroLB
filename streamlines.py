import numpy as np
import matplotlib.pyplot as plt
from util import *

v = np.load("output/BaseLattice_last_u.npy")
obstacle = generate_obstacle_tensor("input/pdrop/pdrop3_60deg1.png").cpu().numpy()
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
plt.streamplot(X, Y, U, V, density=2, linewidth=.1, arrowstyle='-', color=magnitude, cmap=cmap, broken_streamlines=False)

plt.savefig('output/test_3.png', dpi=1500)
# plt.show()
