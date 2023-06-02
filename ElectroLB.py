import matplotlib.pyplot as plt
import torch
import threading
import queue
from PIL import Image
import numpy as np
from alive_progress import alive_bar
import time
from util import *
# To Generate ffmpeg video from images
# ffmpeg -f image2 -framerate 30 -i %05d.png -s 1080x720 output.mp4

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""Simulation parameters"""
# Create obstacle tensor from numpy array
obstacle = generate_obstacle_tensor('input/halfcell.png')
obstacle = obstacle.clone().to(device)
nx, ny = obstacle.shape  # Number of nodes in x and y directions
re = 200  # Reynolds number
ulb = 0.04  # characteristic velocity (inlet)
nulb = ulb * ny / re  # kinematic viscosity
omega = 1 / (3 * nulb + 0.5)  # relaxation parameter


def equilibrium():
    global feq
    # Calculate equilibrium populations (Kruger et al., page 64)
    usqr = 3 / 2 * (u[0] ** 2 + u[1] ** 2)
    cu = 3 * torch.einsum('ixy,ji->jxy', u, c)  # previously ijk,li->ljk
    feq = rho * w.view(9, 1, 1) * (1 + cu + 0.5 * cu ** 2 - usqr)


# Initialize macroscopic variables
rho = torch.ones((nx, ny), device=device).float()
u = torch.zeros((2, nx, ny), device=device).float()

# Initialize populations
feq = torch.zeros((9, nx, ny), device=device).float()
equilibrium()  # Initialize equilibrium populations
fin = feq.clone()  # Initialize incoming populations (pre-collision)
fout = feq.clone()  # Initialize outgoing populations (post-collision)


def macroscopic():
    global rho, u
    # Calculate macroscopic variables rho and u (Kruger et al., page 63)
    rho = fin.sum(0)  # Sum along first axis (populations in each node)
    u = torch.einsum('ji,jxy->ixy', c, fin) / rho


def stream():
    """
    6---2---5
    | \ | / |
    3---0---1
    | / | \ |
    7---4---8
    """
    # Streaming periodically
    global nx, ny, fin, fout
    fin[1, 1:, :] = fout[1, :nx - 1, :]  # vel 1 increases x
    fin[1, 0, :] = fout[1, -1, :]  # wrap
    fin[3, :nx - 1, :] = fout[3, 1:, :]  # vel 3 decreases x
    fin[3, -1, :] = fout[3, 0, :]  # wrap

    fin[2, :, 1:] = fout[2, :, :ny - 1]  # vel 2 increases y
    fin[2, :, 0] = fout[2, :, -1]  # wrap
    fin[4, :, :ny - 1] = fout[4, :, 1:]  # vel 4 decreases y
    fin[4, :, -1] = fout[4, :, 0]  # wrap

    # vel 5 increases x and y simultaneously
    fin[5, 1:, 1:] = fout[5, :nx - 1, :ny - 1]
    fin[5, 0, :] = fout[5, -1, :]  # wrap right
    fin[5, :, 0] = fout[5, :, -1]  # wrap top
    # vel 7 decreases x and y simultaneously
    fin[7, :nx - 1, :ny - 1] = fout[7, 1:, 1:]
    fin[7, -1, :] = fout[7, 0, :]  # wrap left
    fin[7, :, -1] = fout[7, :, 0]  # wrap bottom

    # vel 6 decreases x and increases y
    fin[6, :nx - 1, 1:] = fout[6, 1:, :ny - 1]
    fin[6, -1, :] = fout[6, 0, :]  # wrap left
    fin[6, :, 0] = fout[6, :, -1]  # wrap top
    # vel 8 increases x and decreases y
    fin[8, 1:, :ny - 1] = fout[8, :nx - 1, 1:]
    fin[8, 0, :] = fout[8, -1, :]  # wrap right
    fin[8, :, -1] = fout[8, :, 0]  # wrap bottom

    fin[0, :, :] = fout[0, :, :]  # vel 0 is stationary (dont act like you didn't forget this for 2 hours)


def step():
    global fin, fout, rho, u
    # Perform one LBM step
    # Outlet BC
    # Doing this first is more stable for some reason
    fin[left_col, -1, :] = fin[left_col, -2, :]

    macroscopic()  # Calculate macroscopic variables
    # Impose conditions on macroscopic variables
    u[0, 0, :] = ulb * torch.ones(ny, device=device).float()
    rho[0, :] = 1 / (1 - u[0, 0, :]) * (torch.sum(fin[center_col, 0, :], dim=0) +
                                                  2 * torch.sum(fin[left_col, 0, :], dim=0))

    # Equilibrium
    equilibrium()

    # Boundary conditions on populations
    # Zou-He BC Fin = Feq + Fin(op) - Feq(op)
    fin[right_col, 0, :] = feq[right_col, 0, :] + fin[left_col, 0, :] - feq[left_col, 0, :]

    # BGK collision
    fout = fin - omega * (fin - feq)

    # Bounce-back
    fout[:, obstacle] = fin[c_op][:, obstacle]

    # Streaming
    stream()


def run(iterations: int, save_to_disk: bool = True, interval: int = 100):
    # Launches LBM simulation and a parallel thread for saving data to disk

    if save_to_disk:
        # Create queue for saving data to disk
        q = queue.Queue()
        # Create thread for saving data
        t = threading.Thread(target=save_data, args=(q,))
        t.start()

    # Run LBM for specified number of iterations
    with alive_bar(iterations) as bar:
        start = time.time()
        counter = 0
        for i in range(iterations):
            step()  # Perform one LBM step
            if i % interval == 0:
                # Calculate MLUPS by dividing number of nodes by time in seconds
                dt = time.time() - start
                mlups = nx * ny * counter / (dt * 1e6)
                if save_to_disk:
                    # push data to queue
                    q.put((u, f"output/{i // interval:05}.png"))  # Five digit filename
                # Reset timer and counter
                start = time.time()
                counter = 0

            counter += 1
            bar.text(f"MLUPS: {mlups:.2f}")
            bar()

    # Save final data to numpy files
    np.save(f"output/BaseLattice_last_u.npy", u.cpu().numpy())
    np.save(f"output/BaseLattice_last_rho.npy", rho.cpu().numpy())

    if save_to_disk:
        # Stop thread for saving data
        q.put((None, None))
        t.join()


def save_data(q: queue.Queue):
    # Save data to disk by running a separate thread that gets data from a queue
    while True:
        data, filename = q.get()
        if data is None:
            break
        plt.clf()
        plt.axis('off')
        usqr = data[0] ** 2 + data[1] ** 2
        usqr[obstacle] = np.nan
        plt.imshow(np.sqrt(usqr.cpu().numpy().transpose()), cmap=cmap)
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=500)


if __name__ == '__main__':
    run(10000, save_to_disk=True)