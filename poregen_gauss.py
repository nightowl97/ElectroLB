import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.signal import convolve2d
from PIL import Image
from util import *

cmap = matplotlib.colormaps['Greys']
display_interface = False  # Colors Electrodes sfc in blue for identification in LBM obstacle

nx, ny = 299, 299  # domain dimensions
target_rho = 0.8


# Custom gaussian kernel with multivariate normal distribution and custom covariance matrix
def gaussian_kernel(size: int, mean: float, cov: np.ndarray, angle: float = 0) -> np.ndarray:
    """Creates a 2D Gaussian kernel with given parameters."""
    # Create 2D coordinates
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    x, y = np.meshgrid(x, y)

    # Mean vector
    mean_vec = np.array([0, 0])

    # Create the rotation matrix
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle),  np.cos(angle)]])

    # Rotate the covariance matrix
    cov_rotated = rotation_matrix @ cov @ rotation_matrix.T

    # Create multivariate normal distribution
    dist = multivariate_normal(mean_vec, cov_rotated)

    # Evaluate the PDF on the grid and reshape it into 2D
    kernel = dist.pdf(np.dstack([x, y]))

    return kernel


# kernel parameters
size = 101  # Size of the kernel. Should be odd, to have a center pixel
mean = 0   # Mean. Should be 0 for a centered kernel
cov = np.asarray([[1, 0],
                  [0, 1]])  # Covariance matrix

# Generate the kernel
kernel = gaussian_kernel(size, mean, .1 * cov, angle=torch.pi / 2)

plt.imshow(kernel, interpolation='none')
plt.axis('off')
# plt.savefig(f'output/gaussian_kernel_piby2.png', bbox_inches='tight', pad_inches=0, dpi=600)
plt.show()
# exit()
# Generate random noise matrix from normal distribution
noise = np.random.randn(nx, ny)

print("Applying Gaussian Smoothing..")
# Smooth it with gaussian kernel
smooth = convolve2d(noise, kernel, mode='same', boundary='symm')

# Calculate cutoff for given target porosity using percent point function
cutoff = norm.ppf(target_rho, loc=np.mean(smooth), scale=np.std(smooth))

# Generate boolean matrix
final = smooth > cutoff
print(f'Porosity: {1 - np.sum(final) / (nx * ny)}')

# plt.imshow(final.T, cmap=cmap, interpolation='none')
# plt.axis('off')
# plt.savefig(f'output/electrode.png', bbox_inches='tight', pad_inches=0, dpi=800)
# plt.show()

interface = np.full_like(final, False)
# # identify interface
print("Finding interface..")
for i in range(1, nx - 1):
    for j in range(1, ny - 1):
        neighborhood = final[i - 1: i + 1, j - 1: j + 1]
        if not (neighborhood.all() or not neighborhood.any()):
            interface[i, j] = True

# Obstacles on Boundaries are considered electrode surface
interface[0, final[0]] = True
interface[-1, final[-1]] = True
# interface[final[:, 0], 0] = True
# interface[final[:, -1], -1] = True

print("Drawing Image..")
rgb_array = 255 * np.ones((nx, ny, 3))
rgb_array[final, :] = np.asarray(BLACK)
rgb_array[interface, :] = np.asarray(BLUE)
image = Image.fromarray(rgb_array.transpose(1, 0, 2).astype(np.uint8), mode='RGB')
image.save(f'output/temp.png')
print("Done.")
