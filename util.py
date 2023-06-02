import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import queue

BLACK = np.asarray([0, 0, 0])
WHITE = np.asarray([255, 255, 255])
RED = np.asarray([255, 0, 0])
GREEN = np.asarray([0, 255, 0])
BLUE = np.asarray([0, 0, 255])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cmap = plt.get_cmap('coolwarm')
cmap.set_bad((0, 0, 0, 1))

"""
Base lattice that handles the basic logic of LBM for the Navier-Stokes equations
Outgoing directions:
6---2---5
| \ | / |
3---0---1
| / | \ |
7---4---8
Lattice parameters:
"""
c = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
                 device=device).float()
c_op = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], device=device)  # Opposite directions indices
w = torch.tensor([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], device=device)  # weights
right_col = [1, 5, 8]  # Right column of velocities
left_col = [3, 7, 6]  # Left column of velocities (order is important, see line 85 equilibrium function in inlet)
center_col = [0, 2, 4]  # Center column of velocities


def generate_obstacle_tensor(file):
    # Generate obstacle tensor from image file
    img_array = np.asarray(Image.open(file))
    # Black pixels are True, white pixels are False

    obstacle_solid = (img_array == BLACK).all(axis=2).T
    obstacle_electrode = (img_array == BLUE).all(axis=2).T
    obstacle = torch.tensor(obstacle_solid | obstacle_electrode, dtype=torch.bool).to(device)
    return obstacle


def generate_electrode_tensor(file):
    # Generate electrode tensor from image file
    img_array = np.asarray(Image.open(file))
    # Black pixels are True, white pixels are False

    electrode = (img_array == BLUE).all(axis=2).T
    electrode = torch.tensor(electrode, dtype=torch.bool).to(device)
    return electrode
