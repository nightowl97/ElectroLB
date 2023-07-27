import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from numba import njit
from concurrent.futures import ProcessPoolExecutor
import matplotlib
import joblib

# Note: Emulate this script on terminal to avoid pickling errors with the multiprocessing module

# use interactive backend for matplotlib
matplotlib.use('TkAgg')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"
})
cmap = plt.get_cmap('coolwarm')

nx, ny = 1000, 1000
noise = np.random.randn(nx, ny)

sigmas = np.linspace(3, 40, 38)
rhos = np.linspace(0.1, 0.9, 9)

@njit
def calc_interface(final):
    interface = np.full_like(final, True)
    for x in range(1, final.shape[0] - 1):
        for y in range(1, final.shape[1] - 1):
            tot = final[x - 1: x + 1, y - 1: y + 1]
            if tot.all() or not tot.any():
                interface[x, y] = False
    return interface


def worker(sigma_rho_index):
    i, sigma, j, rho = sigma_rho_index
    print(f"Sigma is:\t {sigma}")
    print(f"Porosity is:\t {rho}")
    # Smooth it with gaussian kernel
    smooth = gaussian_filter(noise, sigma=sigma)
    # Threshold the image
    cutoff = norm.ppf(rho, loc=np.mean(smooth), scale=np.std(smooth))
    final = smooth > cutoff
    # Calculate the interface surface from the binary image
    interface = calc_interface(final)
    return (i, j, np.sum(interface))


def main():
    # Calculate sfc in parallel using multiprocessing
    sfc_indices = [(i, sigma, j, rho) for i, sigma in enumerate(sigmas) for j, rho in enumerate(rhos)]
    sfc = np.zeros((len(sigmas), len(rhos)))

    with ProcessPoolExecutor() as executor:
        for i, j, sum_interface in executor.map(worker, sfc_indices):
            sfc[i, j] = sum_interface

    # 3D plot of the surface as a function of sigma and rho
    X, Y = np.meshgrid(rhos, sigmas)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_ylabel(r'Standard Deviation $\sigma$')
    ax.set_xlabel(r'Porosity $\rho$')
    ax.set_zlabel(r'Normalized surface area')
    sfc /= np.max(sfc)
    surf = ax.plot_surface(X, Y, sfc, cmap=cmap)

    # line
    x = 0.5
    y = np.linspace(2.8, 40, 100)
    z = 3 / y
    # stack into array in the form of [[x, y, z], [x, y, z], ...]
    line = np.stack((x * np.ones_like(y), y, z), axis=1)
    ax.plot(line[:, 0], line[:, 1], line[:, 2], color='black', linewidth=2, linestyle='--',
            label=r"$z(\sigma) = \frac{3}{\sigma}$", zorder=5)
    ax.azim = 30
    ax.elev = 10

    # legend
    ax.legend(loc='upper right', frameon=False, fontsize='x-large')
    # plt.show()
    # return
    plt.savefig('tortparam.png', dpi=900)


if __name__ == '__main__':
    main()
