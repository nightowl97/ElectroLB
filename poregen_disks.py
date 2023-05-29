import matplotlib.pyplot as plt
import numpy as np
cmap = plt.get_cmap('Greys')

nx, ny = 1400, 1400
target_porosity = 0.7
# Empty boolean array
D = np.zeros((nx, ny), dtype=bool)


def add_disk(x, y, r):
    for i in range(nx):
        for j in range(ny):
            if (i - x)**2 + (j - y)**2 < r**2:
                D[i, j] = True


current_porosity = 1

while current_porosity > target_porosity:
    x = np.random.randint(0, nx)
    y = np.random.randint(0, ny)
    r = np.random.normal(50, 5)
    add_disk(x, y, r)

    # recalculate porosity
    current_porosity = 1 - np.sum(D) / (nx * ny)
    print(current_porosity)
    # plt.imshow(D)
    # plt.show()

plt.imshow(D, cmap=cmap)
plt.axis('off')
plt.savefig(f'output/disk_rho_{target_porosity}.png', bbox_inches='tight', pad_inches=0, dpi=600)
# plt.show()