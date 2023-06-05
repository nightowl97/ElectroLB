import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.signal import convolve2d

cmap = matplotlib.colormaps['Greys']

nx, ny = 1000, 1000
target_rho = 0.5


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
size = 201  # Size of the kernel. Should be odd, to have a center pixel
mean = 0   # Mean. Should be 0 for a centered kernel
cov = np.asarray([[1, 0],
                  [0, 10]])  # Covariance matrix

# Generate the kernel
kernel = gaussian_kernel(size, mean, .1 * cov, angle=np.pi / 4)

plt.imshow(kernel, interpolation='none')
plt.axis('off')
plt.savefig(f'output/gaussian_kernel_piby2.png', bbox_inches='tight', pad_inches=0, dpi=600)
# exit()
# Generate random noise matrix from normal distribution
noise = np.random.randn(nx, ny)

# Smooth it with gaussian kernel
smooth = convolve2d(noise, kernel, mode='same', boundary='symm')

# Calculate cutoff for given target porosity using percent point function
cutoff = norm.ppf(target_rho, loc=np.mean(smooth), scale=np.std(smooth))

# Generate boolean matrix
final = smooth > cutoff

plt.imshow(final, cmap=cmap, interpolation='none')
plt.axis('off')
plt.savefig(f'output/anisotropic.png', bbox_inches='tight', pad_inches=0, dpi=600)
# plt.show()
