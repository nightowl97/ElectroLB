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
# ffmpeg -f image2 -framerate 30 -i %05d.png -s 1080x720 -pix_fmt yuv420p output.mp4

"""Simulation parameters"""
# Create obstacle tensor from numpy array
obstacle = generate_obstacle_tensor('input/input2.png')
obstacle = obstacle.clone().to(device)
nx, ny = obstacle.shape  # Number of nodes in x and y directions
re = 100  # Reynolds number
ulb = 0.04  # characteristic velocity (inlet)
nulb = ulb * ny / re  # kinematic viscosity
omega = 1 / (3 * nulb + 0.5)  # relaxation parameter
omega_f = omega_g = omega  # TODO: omega_f and omega_g can be different
print(f"omega: {omega}")


def equilibrium():
    global feq, geq
    # Kruger et al., page 382, equation 9.125
    u_eq_f = u_p + fsc_f / (omega_f * rho_f)
    u_eq_g = u_p + fsc_g / (omega_g * rho_g)
    # Calculate equilibrium populations (Kruger et al., page 64)
    usqr_f = 3 / 2 * (u_eq_f[0] ** 2 + u_eq_f[1] ** 2)
    usqr_g = 3 / 2 * (u_eq_g[0] ** 2 + u_eq_g[1] ** 2)

    cu_f = 3 * torch.einsum('ixy,ji->jxy', u_eq_f, c)  # previously ijk,li->ljk
    cu_g = 3 * torch.einsum('ixy,ji->jxy', u_eq_g, c)  # previously ijk,li->ljk
    feq = rho_f * w.view(9, 1, 1) * (1 + cu_f + 0.5 * cu_f ** 2 - usqr_f)
    geq = rho_g * w.view(9, 1, 1) * (1 + cu_g + 0.5 * cu_g ** 2 - usqr_g)


# Shan chen forces
fsc_f = torch.zeros((2, nx, ny), device=device).float()
fsc_g = torch.zeros((2, nx, ny), device=device).float()
# Initialize macroscopic variables
rho_f = torch.ones((nx, ny), device=device).float() + 0.1 * torch.rand((nx, ny), device=device).float()
rho_g = torch.ones((nx, ny), device=device).float() + 0.1 * torch.rand((nx, ny), device=device).float()
u_p = torch.zeros((2, nx, ny), device=device).float()
u_f = torch.zeros((2, nx, ny), device=device).float()
u_g = torch.zeros((2, nx, ny), device=device).float()

# Initialize populations
feq = torch.zeros((9, nx, ny), device=device).float()
geq = torch.zeros((9, nx, ny), device=device).float()
equilibrium()  # Initialize equilibrium populations
fin = feq.clone()  # Initialize incoming populations (pre-collision)
gin = geq.clone()
fout = feq.clone()  # Initialize outgoing populations (post-collision)
gout = geq.clone()


def shan_chen_force():
    global fsc_f, fsc_g, rho_f, rho_g
    G = -3  # interaction strength (negative for repulsion)
    # Pseudopotential
    psi_f = 1 - torch.exp(-rho_f)
    psi_g = 1 - torch.exp(-rho_g)
    # Calculate shan-chen forces
    fsc_f[:, :, :] = 0
    fsc_g[:, :, :] = 0
    # Vel 1 pushes from right
    fsc_f[0, :-1, :] += (w[1] * psi_g[1:, :] * (c[1, 0]))
    fsc_g[0, :-1, :] += (w[1] * psi_f[1:, :] * (c[1, 0]))

    # Vel 3 pushes from left
    fsc_f[0, 1:, :] += (w[3] * psi_g[:-1, :] * (c[3, 0]))
    fsc_g[0, 1:, :] += (w[3] * psi_f[:-1, :] * (c[3, 0]))

    # Vel 2 pushes from top
    fsc_f[1, :, :-1] += (w[2] * psi_g[:, 1:] * c[2, 1])
    fsc_g[1, :, :-1] += (w[2] * psi_f[:, 1:] * c[2, 1])

    # Vel 4 pushes from bottom
    fsc_f[1, :, 1:] += (w[4] * psi_g[:, :-1] * c[4, 1])
    fsc_g[1, :, 1:] += (w[4] * psi_f[:, :-1] * c[4, 1])

    # Vel 5 pushes from top right
    fsc_f[0, :-1, :-1] += (w[5] * psi_g[1:, 1:] * c[5, 0])
    fsc_f[1, :-1, :-1] += (w[5] * psi_g[1:, 1:] * c[5, 1])
    fsc_g[0, :-1, :-1] += (w[5] * psi_f[1:, 1:] * c[5, 0])
    fsc_g[1, :-1, :-1] += (w[5] * psi_f[1:, 1:] * c[5, 1])

    # Vel 6 pushes from top left
    fsc_f[0, 1:, :-1] += (w[6] * psi_g[:-1, 1:] * c[6, 0])
    fsc_f[1, 1:, :-1] += (w[6] * psi_g[:-1, 1:] * c[6, 1])
    fsc_g[0, 1:, :-1] += (w[6] * psi_f[:-1, 1:] * c[6, 0])
    fsc_g[1, 1:, :-1] += (w[6] * psi_f[:-1, 1:] * c[6, 1])

    # Vel 7 pushes from bottom left
    fsc_f[0, 1:, 1:] += (w[7] * psi_g[:-1, :-1] * c[7, 0])
    fsc_f[1, 1:, 1:] += (w[7] * psi_g[:-1, :-1] * c[7, 1])
    fsc_g[0, 1:, 1:] += (w[7] * psi_f[:-1, :-1] * c[7, 0])
    fsc_g[1, 1:, 1:] += (w[7] * psi_f[:-1, :-1] * c[7, 1])

    # Vel 8 pushes from bottom right
    fsc_f[0, :-1, 1:] += (w[8] * psi_g[1:, :-1] * c[8, 0])
    fsc_f[1, :-1, 1:] += (w[8] * psi_g[1:, :-1] * c[8, 1])
    fsc_g[0, :-1, 1:] += (w[8] * psi_f[1:, :-1] * c[8, 0])
    fsc_g[1, :-1, 1:] += (w[8] * psi_f[1:, :-1] * c[8, 1])

    # periodic boundaries
    # left boundary
    # fsc_f[0, 0, :] += (w[1] * rho_g[-1, :] * c[1, 0]) * rho_f[0, :]
    # fsc_g[0, 0, :] += (w[1] * rho_f[-1, :] * c[1, 0]) * rho_g[0, :]
    # # right boundary
    # fsc_f[0, -1, :] += (w[3] * rho_g[0, :] * c[3, 0]) * rho_f[-1, :]
    # fsc_g[0, -1, :] += (w[3] * rho_f[0, :] * c[3, 0]) * rho_g[-1, :]
    # # bottom boundary
    # fsc_f[1, :, 0] += (w[2] * rho_g[:, -1] * c[2, 1]) * rho_f[:, 0]
    # fsc_g[1, :, 0] += (w[2] * rho_f[:, -1] * c[2, 1]) * rho_g[:, 0]
    # # top boundary
    # fsc_f[1, :, -1] += (w[4] * rho_g[:, 0] * c[4, 1]) * rho_f[:, -1]
    # fsc_g[1, :, -1] += (w[4] * rho_f[:, 0] * c[4, 1]) * rho_g[:, -1]

    fsc_f *= G * psi_f
    fsc_g *= G * psi_g


def macroscopic():
    global rho_f, rho_g, u_f, u_g, u_p
    # Calculate macroscopic variables rho and u (Kruger et al., page 63)
    rho_f = fin.sum(0)  # Sum along first axis (populations in each node)
    rho_g = gin.sum(0)
    # no barycentric velocity for Guo forcing (Kruger et al., page 381-382)
    u_f = (torch.einsum('ji,jxy->ixy', c, fin))
    u_g = (torch.einsum('ji,jxy->ixy', c, gin))
    # kruger et al., page 382, equation (9.126)
    u_p = (u_f * omega_f + u_g * omega_g) / (rho_f * omega_f + rho_g * omega_g)


def stream(fin, fout):
    """
    6---2---5
    | \ | / |
    3---0---1
    | / | \ |
    7---4---8
    """
    # Streaming periodically
    global nx, ny
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
    global fin, fout, gin, gout, rho_f, rho_g, u_f, u_g, u_p
    # Perform one LBM step
    # Outlet BC
    # fin[left_col, -1, :] = fin[left_col, -2, :]
    # gin[left_col, -1, :] = gin[left_col, -2, :]

    macroscopic()  # Calculate macroscopic variables
    shan_chen_force()  # Calculate Shan-Chen force
    # Impose conditions on macroscopic variables
    u_p[0, 0, :] = ulb * torch.ones(ny, device=device).float()
    # u_g[0, 0, :] = ulb * torch.ones(ny, device=device).float()
    rho_f[0, :ny//2] = 1 / (1 - u_p[0, 0, :ny//2]) * (torch.sum(fin[center_col, 0, :ny//2], dim=0) +
                                                  2 * torch.sum(fin[left_col, 0, :ny//2], dim=0))
    rho_f[0, ny//2:] = 0.1
    rho_g[0, ny//2:] = 1 / (1 - u_p[0, 0, ny//2:]) * (torch.sum(gin[center_col, 0, ny//2:], dim=0) +
                                                  2 * torch.sum(gin[left_col, 0, ny//2:], dim=0))
    rho_g[0, :ny//2] = 0.1

    # Equilibrium
    equilibrium()

    # Boundary conditions on populations
    # Zou-He BC Fin = Feq + Fin(op) - Feq(op)
    fin[right_col, 0, :] = feq[right_col, 0, :] + fin[left_col, 0, :] - feq[left_col, 0, :]
    gin[right_col, 0, :] = geq[right_col, 0, :] + gin[left_col, 0, :] - geq[left_col, 0, :]

    # BGK collision
    fout = fin - omega_f * (fin - feq)
    gout = gin - omega_g * (gin - geq)

    # Bounce-back
    fout[:, obstacle] = fin[c_op][:, obstacle]
    gout[:, obstacle] = gin[c_op][:, obstacle]

    # Streaming
    stream(fin, fout)
    stream(gin, gout)


def run(iterations: int, save_to_disk: bool = True, interval: int = 100, continue_last: bool = False):
    # Launches LBM simulation and a parallel thread for saving data to disk
    global rho_f, rho_g, u_p, fin, fout, gin, gout
    if continue_last:  # Continue last computation
        rho_f = torch.from_numpy(np.load("output/BaseLattice_last_rho_f.npy")).to(device)
        rho_g = torch.from_numpy(np.load("output/BaseLattice_last_rho_g.npy")).to(device)
        u_p = torch.from_numpy(np.load("output/BaseLattice_last_u.npy")).to(device)
        equilibrium()
        fin = feq.clone()  # Initialize incoming populations (pre-collision)
        fout = feq.clone()  # Initialize outgoing populations (post-collision)
        gin = geq.clone()
        gout = geq.clone()

    if save_to_disk:
        # Create queue for saving data to disk
        q = queue.Queue()
        # Create thread for saving data
        t = threading.Thread(target=save_data, args=(q, ulb, obstacle))
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
                    q.put(((u_p.clone(), rho_f.clone()), f"output/{i // interval:05}.png"))  # Five digit filename
                # Reset timer and counter
                start = time.time()
                counter = 0

            counter += 1
            bar.text(f"MLUPS: {mlups:.2f}")
            bar()

    # Save final data to numpy files
    np.save(f"output/BaseLattice_last_u.npy", u_p.cpu().numpy())
    np.save(f"output/BaseLattice_last_rho_f.npy", rho_f.cpu().numpy())
    np.save(f"output/BaseLattice_last_rho_g.npy", rho_g.cpu().numpy())

    if save_to_disk:
        # Stop thread for saving data
        q.put((None, None))
        t.join()


if __name__ == '__main__':
    print("Using device: ", device)
    run(50000, save_to_disk=True, interval=100, continue_last=False)