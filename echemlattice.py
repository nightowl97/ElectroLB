import numpy as np
import torch
from PIL import Image
from util import *
import threading
import queue
from alive_progress import alive_bar
import time

# Solves the convective-diffusion equation for a species in an electrochemical system
T = 298  # Kelvin
R = 8.3145  # J/mol.K
F = 96485  # C/mol
z = 1  # Number of electrons transferred
E_0 = 0.6  # Standard potential
alpha = 0.65e-5  # Diffusion coefficient (Bard page 1013)
j0 = 1e-1  # Exchange current density

electrode = generate_electrode_tensor("input/ecell_small.png")
obstacle = generate_obstacle_tensor("input/ecell_small.png")
v_field = torch.from_numpy(np.load('output/BaseLattice_last_u.npy'))
v_field = v_field.clone().to(device)  # Velocity field

v_field[:, :, :] = 0  # temporary


"""Simulation parameters"""
# Diffusion coefficient
d = .2
tau = 3 * d + 0.5  # Relaxation time
omega = 1 / tau  # TODO: add independent omega for each species
nx, ny = obstacle.shape  # Number of nodes in x and y directions

"""Initialization"""
# Initialize scalar field for species concentration
rho_ox = torch.ones((nx, ny), device=device)
rho_red = torch.ones((nx, ny), device=device)
rho_ox[:, 1] = 1  # Inlet concentration
rho_red[:, 1] = 1  # Inlet concentration

feq_ox = torch.zeros((9, nx, ny), device=device)
feq_red = torch.zeros((9, nx, ny), device=device)


def equilibrium():
    global feq_ox, feq_red
    # Calculate equilibrium populations (Kruger page 304)
    cu = torch.einsum('ixy,ji->jxy', v_field, c)  # TODO: Precalculate if v_field is constant
    feq_ox = w.view(9, 1, 1) * rho_ox * (1 + 3 * cu)
    feq_red = w.view(9, 1, 1) * rho_red * (1 + 3 * cu)


equilibrium()

fin_ox = feq_ox.clone()
fin_red = feq_red.clone()

fout_ox = feq_ox.clone()
fout_red = feq_red.clone()

source_ox = torch.zeros_like(rho_ox, device=device)
source_red = torch.zeros_like(rho_red, device=device)

# Set electrode potential
e = E_0 * torch.ones(10000, device=device)
# Nernst potential on electrode verify if needed R * T / (z * F)
e_nernst = torch.ones_like(electrode, device=device) * E_0
# Current density
j = torch.zeros_like(e_nernst, device=device)
j_log = torch.empty(10000, device=device)

"""LBM operations"""


def macroscopic():
    global rho_ox, rho_red
    rho_ox = fin_ox.sum(0)
    rho_red = fin_red.sum(0)


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


def step(i):
    global fin_ox, fin_red, fout_ox, fout_red, source_ox, source_red, rho_ox, rho_red, e_nernst, j, e
    # Perform one LBM step
    # Outlet BC
    # Equiv. to neumann BC on concentration (null flux)
    fin_ox[left_col, -1, :] = fin_ox[left_col, -2, :]
    fin_red[left_col, -1, :] = fin_red[left_col, -2, :]

    macroscopic()

    # Inlet BC
    rho_ox[:, ny - 2] = 1  # Inlet concentration
    rho_red[:, ny - 2] = 1  # Inlet concentration

    equilibrium()

    # Zhou He BC
    fin_ox[right_col, 0, :] = feq_ox[right_col, 0, :] + fin_ox[left_col, 0, :] - feq_ox[left_col, 0, :]
    fin_red[right_col, 0, :] = feq_red[right_col, 0, :] + fin_red[left_col, 0, :] - feq_red[left_col, 0, :]

    # Electrode BC
    e_nernst = E_0 + torch.log(rho_ox[electrode] / rho_red[electrode])
    j = j0 * (rho_ox[electrode] * torch.exp(0.5 * (e[i] - e_nernst)) -
                rho_red[electrode] * torch.exp(-0.5 * (e[i] - e_nernst)))

    source_ox[electrode] = -j
    source_red[electrode] = j
    j_log[i] = torch.sum(j)  # Log current density

    # BGK collision
    fout_ox = fin_ox - omega * (fin_ox - feq_ox) + torch.einsum('i,jk->ijk', w, source_ox)
    fout_red = fin_red - omega * (fin_red - feq_red) + torch.einsum('i,jk->ijk', w, source_red)

    # Bounce-back
    fout_ox[:, obstacle] = fin_ox[c_op][:, obstacle]
    fout_red[:, obstacle] = fin_red[c_op][:, obstacle]

    # Streaming
    stream(fin_ox, fout_ox)
    stream(fin_red, fout_red)


def run(iterations: int, save_to_disk: bool = True, interval: int = 100, continue_last: bool = False):
    # Launches LBM simulation and a parallel thread for saving data to disk
    global rho_ox, rho_red, fin_ox, fin_red, fout_ox, fout_red, j_log, e
    j_log = torch.zeros(iterations, device=device)
    e = torch.zeros(iterations, device=device)
    e += (2 * E_0 / iterations) * torch.arange(iterations, device=device)
    if continue_last:  # Continue last computation
        rho_ox = torch.from_numpy(np.load("output/Electrochemical_last_rho_ox.npy")).to(device)
        rho_red = torch.from_numpy(np.load("output/Electrochemical_last_rho_red.npy")).to(device)
        equilibrium()
        fin_ox = feq_ox.clone()  # Initialize incoming populations (pre-collision)
        fin_red = feq_red.clone()  # Initialize incoming populations (pre-collision)
        fout_ox = feq_ox.clone()  # Initialize outgoing populations (post-collision)
        fout_red = feq_red.clone()  # Initialize outgoing populations (post-collision)

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
            step(i)  # Perform one LBM step
            if i % interval == 0:
                # Calculate MLUPS by dividing number of nodes by time in seconds
                dt = time.time() - start
                mlups = nx * ny * counter / (dt * 1e6)
                if save_to_disk:
                    # push data to queue
                    q.put((rho_ox, f"output/{i // interval:05}.png"))  # Five digit filename
                # Reset timer and counter
                start = time.time()
                counter = 0

            counter += 1
            bar.text(f"MLUPS: {mlups:.2f} | Total density {rho_ox.mean().cpu().numpy():.5f}")
            bar()

    # Save final data to numpy files
    np.save(f"output/Electrochemical_last_rho_ox.npy", rho_ox.cpu().numpy())
    np.save(f"output/Electrochemical_last_rho_red.npy", rho_red.cpu().numpy())

    # Plot current density
    fig, ax = plt.subplots()
    ax.plot(e, j_log.cpu().numpy())
    plt.show()
    plt.close(fig)

    if save_to_disk:
        # Stop thread for saving data
        q.put((None, None))
        t.join()


def save_data(q: queue.Queue):
    while True:
        data, filename = q.get()
        if data is None:
            break
        plt.clf()
        plt.axis('off')
        data[obstacle] = np.nan
        plt.imshow(data.cpu().numpy().transpose(), cmap=cmap, vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=500)


if __name__ == '__main__':
    print(f"omega: {omega}")
    run(10000, save_to_disk=True, interval=100, continue_last=False)
