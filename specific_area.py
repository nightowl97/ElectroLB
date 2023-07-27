import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from util import *

inputs = [0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96]
specific_areas = []

for i in inputs:
    # Load image
    img = Image.open(f'input/porosities/electrode_{i}.png')
    img_array = np.array(img)
    # image area
    total_area = img_array.shape[0] * img_array.shape[1]
    solid_area = np.sum((img_array == BLACK).all(axis=2))
    fluid_area = np.sum((img_array == WHITE).all(axis=2))
    sfc_area = np.sum((img_array == BLUE).all(axis=2))

    # print(f"Total area: {total_area}")
    # print(f"Solid area: {solid_area}")
    # print(f"Fluid area: {fluid_area}")
    # print(f"Surface area: {sfc_area}")

    # porosity
    porosity = fluid_area / total_area
    specific_area = sfc_area / total_area
    # print(f"Porosity: {porosity}")
    # print(f"Specific area: {specific_area}")
    specific_areas.append(specific_area)

plt.plot(inputs, specific_areas, 'ro')
plt.xlabel('Porosity')
plt.ylabel('Specific Area')
plt.show()