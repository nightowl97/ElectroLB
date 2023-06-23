import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import queue

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"
})

BLACK = np.asarray([0, 0, 0])
WHITE = np.asarray([255, 255, 255])
RED = np.asarray([255, 0, 0])
GREEN = np.asarray([0, 255, 0])
BLUE = np.asarray([0, 0, 255])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cmap = plt.get_cmap('coolwarm')
cmap.set_bad((0, 0, 0, 1))

"""
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

top_row = [2, 6, 5]
center_row = [0, 1, 3]
bottom_row = [4, 8, 7]


def generate_obstacle_tensor(file):
    # Generate obstacle tensor from image file
    img_array = np.asarray(Image.open(file))
    # Black pixels are True, white pixels are False

    obstacle_solid = (img_array == BLACK).all(axis=2).T
    obstacle_electrode = (img_array == BLUE).all(axis=2).T
    obstacle = torch.tensor(obstacle_solid, dtype=torch.bool).to(device)
    return obstacle


def generate_electrode_tensor(file):
    # Generate electrode tensor from image file
    img_array = np.asarray(Image.open(file))
    # Black pixels are True, white pixels are False

    electrode = (img_array == BLUE).all(axis=2).T
    electrode = torch.tensor(electrode, dtype=torch.bool).to(device)
    return electrode


def save_data(q: queue.Queue, inlet_vel, obstacle):
    # Save data to disk by running a separate thread that gets data from a queue
    while True:
        data, filename = q.get()
        if data is None:
            break

        # Preprocessing before plotting
        velocity = torch.sqrt(data[0][0] ** 2 + data[0][1] ** 2)  # module of velocity
        # velocity /= inlet_vel  # normalize
        density = data[1]
        velocity[obstacle] = np.nan
        density[obstacle] = np.nan

        # Plot both macroscopic variables
        fig, (ax0, ax1) = plt.subplots(2, 1)
        cax0 = ax0.imshow(velocity.cpu().numpy().transpose(), cmap=cmap)
        cax1 = ax1.imshow(density.cpu().numpy().transpose(), cmap=cmap, vmin=0, vmax=1.5)
        ax0.set_title(r"lattice velocity $u$")
        ax1.set_title(r"density $\rho$")
        ax0.axis("off")
        ax1.axis("off")
        fig.colorbar(cax0, ax=ax0)
        fig.colorbar(cax1, ax=ax1)
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=600)
        plt.close(fig)
