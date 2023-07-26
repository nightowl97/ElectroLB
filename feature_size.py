import cv2
import numpy as np
from PIL import Image

# Load the image
# (Assuming it's a grayscale image where 0 is black and 255 is white)
# If it's a different kind of binary image you might need to adjust
image = np.array(Image.open("input/pdrop/permeability_0.95_0.005.png"))

image = image[2:-2, 2:-2]  # ignore borders

image_array = (image == [0, 0, 0]).all(axis=2).T

# Perform connected components analysis
num_labels, labels = cv2.connectedComponents(image_array.astype(np.uint8))

# Initialize a list to hold the size of each disc (in pixels)
sizes = []

# Loop through each label
for label in range(1, num_labels):  # we skip label 0 as it is the background
    # Create an image for this disc
    disc = np.where(labels == label, 1, 0)
    # Calculate the size of this disc
    disc_size = np.sum(disc)
    # Append the size to our list
    sizes.append(disc_size)

# Calculate the average size
average_size = np.mean(sizes)

print("Average disc size (in pixels): ", average_size)
average_radius = np.sqrt(average_size / np.pi)
print(f"Average radius in pixels: {average_radius}")