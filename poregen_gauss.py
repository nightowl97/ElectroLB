import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

cmap = matplotlib.colormaps['Greys']

nx, ny = 1000, 1000
target_rho = 0.5

# Generate random noise matrix from normal distribution
noise = np.random.randn(nx, ny)

# Smooth it with gaussian kernel
smooth = gaussian_filter(noise, sigma=10)

# Calculate cutoff for given target porosity using percent point function
cutoff = norm.ppf(target_rho, loc=np.mean(smooth), scale=np.std(smooth))

# Generate boolean matrix
final = smooth > cutoff

plt.imshow(final, cmap=cmap)
plt.axis('off')
plt.show()
