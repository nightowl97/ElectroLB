import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from util import *
import threading
import queue
from alive_progress import alive_bar
import time
from scipy.signal import sawtooth

"""
Solves the convective-diffusion equation for a species in an electrochemical system and performs cyclic voltammetry

# Converting from physical to lattice units we dx_l = 1, dt_l = 1, rho_l = 1, therefore our conversion factors are:
C_rho = concetration_ph
C_t = dt
C_length = dx
in order to convert between two systems we use
physical_quantity = lattice_quantity * C_quantity (Kruger page 272)
for example the conversion factor for velocity C_u is C_length / C_t = dx / dt
"""

# Physical constants
cell_size_ph = 5e-2  # 5cm or 0.05m
cell_depth_ph = 5e-3  # 5mm or 0.005m
concentration_ph = 100  # mol/m^3 or 0.1M

z = 1  # Number of electrons transferred
E_0 = 0.6  # Standard potential
d_ph = 0.76e-9  # m^2/s Diffusion coefficient (Bard page 813)
j_0 = 10  # Exchange current density in A/m^2
j_0_ph = 5 * 10000  # A/m2 from lvov page 129 for FerriFerro

electrode = generate_electrode_tensor("input/cottrell.png")
obstacle = generate_obstacle_tensor("input/cottrell.png")
nx, ny = obstacle.shape
v_field = torch.zeros((2, nx, ny), device=device)

"""Simulation parameters"""
# Diffusion coefficient
omega_l = 1.98
fo, dx, dt, d_l = convert_from_physical_params_pure_diff(cell_size_ph, cell_size_ph, d_ph, nx, omega_l)
tau = 1 / omega_l
# j0_l = j0 / dx * cell_depth_ph  # A/lattice_site
input("Press enter to start...")

"""Initialization"""
# Initialize scalar field for species concentration
rho_ox = torch.ones((nx, ny), dtype=torch.float64, device=device)
rho_red = torch.ones((nx, ny), dtype=torch.float64, device=device)
rho_ox[:, 1] = 1  # Inlet concentration
rho_red[:, 1] = 1  # Inlet concentration

feq_ox = torch.zeros((9, nx, ny), dtype=torch.float64, device=device)
feq_red = torch.zeros((9, nx, ny), dtype=torch.float64, device=device)


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

source_ox = torch.zeros_like(rho_ox, dtype=torch.float64, device=device)
source_red = torch.zeros_like(rho_red, dtype=torch.float64, device=device)

# Set electrode potential
e = E_0 * torch.ones(10000, device=device)
# Nernst potential on electrode verify if needed R * T / (z * F)
e_nernst = torch.ones_like(electrode, dtype=torch.float64, device=device) * E_0
# Current density
j = torch.zeros_like(e_nernst, dtype=torch.float64, device=device)
j_log = torch.empty(10000, dtype=torch.float64, device=device)

"""LBM operations"""


def macroscopic():
    global rho_ox, rho_red, source_ox, source_red
    # rho_ox = torch.clamp(fin_ox.sum(0) + source_ox / 2, 0, 2)
    rho_ox = fin_ox.sum(0) + source_ox / 2
    # rho_red = torch.clamp(fin_red.sum(0) + source_red / 2, 0, 2)
    rho_red = fin_red.sum(0) + source_red / 2


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


def step(i, rho_ox_log, src_log):
    global fin_ox, fin_red, fout_ox, fout_red, source_ox, source_red, rho_ox, rho_red, e_nernst, j, e
    # Perform one LBM step
    # Outlet BC
    # Equiv. to neumann BC on concentration (null flux)
    fin_ox[left_col, -1, :] = fin_ox[left_col, -2, :]
    fin_red[left_col, -1, :] = fin_red[left_col, -2, :]

    macroscopic()
    rho_ox_log[i] = torch.mean(rho_ox[electrode])

    # Inlet BC
    rho_ox[:, nx - 1] = 1  # Inlet concentration
    rho_red[:, nx - 1] = 1  # Inlet concentration

    equilibrium()

    # Zhou He BC
    fin_ox[right_col, 0, :] = feq_ox[right_col, 0, :] + fin_ox[left_col, 0, :] - feq_ox[left_col, 0, :]
    fin_red[right_col, 0, :] = feq_red[right_col, 0, :] + fin_red[left_col, 0, :] - feq_red[left_col, 0, :]

    # Electrode BC
    # TODO: Use Tafel in logarithmic space to avoid instabilities
    e_nernst = E_0 - (R * T) / (z * F) * torch.log(torch.mean(rho_red[electrode]) / torch.mean(rho_ox[electrode]))
    # e_nernst = torch.tensor(E_0, device=device)
    # avoid large exponents
    exponent = (.5 * F / (R * T)) * (e[i] - e_nernst)
    # Current density in amps per m^2 in each lbm box
    j = j_0 * concentration_ph * (torch.mean(rho_ox[electrode]) ** .5) * (torch.mean(rho_red[electrode]) ** .5) * \
        (torch.exp(exponent) - torch.exp(-exponent))
    # current in amps per lbm box
    current = torch.nan_to_num(j * dx * cell_depth_ph, nan=0.0)

    charge = current * dt  # Charge per site (C / site)
    matter = charge / (z * F)  # substance created/consumed per site (mol / site)
    c_ph = matter / (dx**2 * cell_depth_ph)  # substance in mol per lbm box
    # substance in lattice units source_ph = C x source_lattice where C is C_rho / C_t
    # Source term units are mol litre^-1 s^-1
    c_l = c_ph * dt / concentration_ph
    # c_l[c_l > rho_ox[electrode]] = 1 * rho_ox[electrode][c_l > rho_ox[electrode]]
    # c_l[-c_l > rho_red[electrode]] = -1 * rho_red[electrode][-c_l > rho_red[electrode]]

    # current = c_l * (concentration_ph * dx**2 * cell_depth_ph * z * F) / (dt**2)
    src_log[i] = torch.mean(c_l)

    source_ox[electrode] = c_l
    source_red[electrode] = -c_l
    j_log[i] = torch.sum(current)  # Log current density

    # BGK collision
    fout_ox = fin_ox - omega_l * (fin_ox - feq_ox) + (1 - 1 / (2 * tau)) * torch.einsum('i,jk->ijk', w, source_ox)
    fout_red = fin_red - omega_l * (fin_red - feq_red) + (1 - 1 / (2 * tau)) * torch.einsum('i,jk->ijk', w, source_red)

    # Bounce-back
    fout_ox[:, obstacle] = fin_ox[c_op][:, obstacle]
    fout_red[:, obstacle] = fin_red[c_op][:, obstacle]

    # Streaming
    stream(fin_ox, fout_ox)
    stream(fin_red, fout_red)


def run(iterations: int, save_to_disk: bool = True, interval: int = 100, continue_last: bool = False):
    # Launches LBM simulation and a parallel thread for saving data to disk
    global rho_ox, rho_red, fin_ox, fin_red, fout_ox, fout_red, j_log, e, dt

    print(f"Simulating {iterations * dt} seconds")

    j_log = torch.zeros(iterations, dtype=torch.float64, device=device)
    rho_ox_log = torch.zeros(iterations, dtype=torch.float64, device=device)
    src_log = torch.zeros(iterations, dtype=torch.float64, device=device)

    buffer_time = 200  # Buffer time for lbm stabilization
    # Set up electrode potential
    t = np.linspace(0, 1, iterations - buffer_time)
    e = torch.ones_like(t) * 1.2 * E_0
    e = torch.cat((torch.zeros(buffer_time), e), dim=0)
    plt.plot(e.cpu().numpy())
    plt.ylabel("Potential (V)")
    plt.xlabel("Iteration")
    plt.grid()
    plt.show()
    plt.close()
    input("Continue..")
    # input("Continue?")
    # print(e[0])
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
    with alive_bar(iterations, force_tty=True) as bar:
        start = time.time()
        counter = 0
        for i in range(iterations):
            step(i, rho_ox_log, src_log)  # Perform one LBM step
            if i % interval == 0:
                # Calculate MLUPS by dividing number of nodes by time in seconds
                delta_t = time.time() - start
                mlups = nx * ny * counter / (delta_t * 1e6)
                if save_to_disk:
                    # push data to queue
                    q.put((rho_ox.detach().clone(), f"output/{i // interval:05}.png"))  # Five digit filename
                # Reset timer and counter
                start = time.time()
                counter = 0

            counter += 1
            bar.text(f"MLUPS: {mlups:.2f} | Electrode density {rho_ox[electrode].mean().cpu().numpy():.5f} | "
                     f"Potential {e[i]:.5f} | Nernst Potential {torch.mean(e_nernst).cpu().numpy():.3f}")
            bar()

    # Save final data to numpy files
    np.save(f"output/Electrochemical_last_rho_ox.npy", rho_ox.cpu().numpy())
    np.save(f"output/Electrochemical_last_rho_red.npy", rho_red.cpu().numpy())

    # Plot current density
    plt.show()
    fig, ax = plt.subplots()
    ax.plot(j_log[buffer_time:].cpu().numpy())
    plt.show()

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
        plt.imshow(data.cpu().numpy().transpose(), cmap=cmap, vmin=0, vmax=2)
        plt.colorbar()
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=500)
        plt.close()


if __name__ == '__main__':
    print(f"omega: {omega_l}")
    run(5000, save_to_disk=True, interval=100, continue_last=False)
