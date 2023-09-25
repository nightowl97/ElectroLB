import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from util import *
import threading
import queue
from alive_progress import alive_bar
import time

# Physical constants
cell_size_ph = 5e-2  # 5cm or 0.05m
cell_depth_ph = 2e-3  # 2mm or 0.002m
concentration_ph = 100  # mol/m^3 or 0.1M
vel_ph = 0.01  # m/s

z = 1  # Number of electrons transferred
d_ph = 0.76e-7  # m^2/s Diffusion coefficient (Bard page 813)


electrode = generate_electrode_tensor("input/leveque_largear.png")
obstacle = generate_obstacle_tensor("input/leveque_largear.png")
nx, ny = obstacle.shape
v_field = torch.from_numpy(np.load("output/BaseLattice_last_u.npy")).to(device)

"""Simulation parameters"""
delay = 200  # Delay before applying voltage
total_iterations = 40000
# Diffusion coefficient
omega_l = 1.95
w_e = omega_l
lambda_trt = 1/12
w_o = 1 / (lambda_trt / (1/w_e - 0.5) + 0.5)
# w_o = 0.6
# lambda_trt = (1/w_e - 0.5) * (1/w_o - 0.5)
pe, dx, dt, d_l, u_l = convert_from_physical_params_diff(cell_size_ph, ny - 2, vel_ph, d_ph, nx, omega_l)
print(f"lambda_trt: {lambda_trt}")
print(f"omega_e: {w_e}, omega_o: {w_o}")
tau = 1 / omega_l
j_log = torch.zeros(1, dtype=torch.float64, device=device)
v_field = v_field / torch.max(v_field) * u_l
input("Press enter to start...")

"""Initialization"""
# Initialize scalar field for species concentration
rho_ox = torch.zeros((nx, ny), dtype=torch.float64, device=device)
rho_red = torch.ones((nx, ny), dtype=torch.float64, device=device)
rho_ox[obstacle] = 0
rho_red[obstacle] = 0

feq_ox = torch.zeros((9, nx, ny), dtype=torch.float64, device=device)
feq_red = torch.zeros((9, nx, ny), dtype=torch.float64, device=device)


def equilibrium():
    global feq_ox, feq_red
    cu = torch.einsum('ixy,ji->jxy', v_field, c)
    # Calculate equilibrium populations (Kruger page 304)
    feq_ox = w.view(9, 1, 1) * rho_ox * (1 + 3 * cu)
    feq_red = w.view(9, 1, 1) * rho_red * (1 + 3 * cu)


equilibrium()

fin_ox = feq_ox.clone()
fin_red = feq_red.clone()

fout_ox = feq_ox.clone()
fout_red = feq_red.clone()

# source_ox = torch.zeros_like(rho_ox, dtype=torch.float64, device=device)
# source_red = torch.zeros_like(rho_red, dtype=torch.float64, device=device)

"""LBM operations"""


def macroscopic():
    global rho_ox, rho_red
    rho_ox = torch.clamp(fin_ox.sum(0), 0, 1)
    # rho_ox = fin_ox.sum(0)  # + source_ox / 2
    rho_red = torch.clamp(fin_red.sum(0), 0, 1)
    # rho_red = fin_red.sum(0)  # + source_red / 2


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
    # fin[1, 0, :] = fout[1, -1, :]  # wrap
    fin[3, :nx - 1, :] = fout[3, 1:, :]  # vel 3 decreases x
    # fin[3, -1, :] = fout[3, 0, :]  # wrap

    fin[2, :, 1:] = fout[2, :, :ny - 1]  # vel 2 increases y
    # fin[2, :, 0] = fout[2, :, -1]  # wrap
    fin[4, :, :ny - 1] = fout[4, :, 1:]  # vel 4 decreases y
    # fin[4, :, -1] = fout[4, :, 0]  # wrap

    # vel 5 increases x and y simultaneously
    fin[5, 1:, 1:] = fout[5, :nx - 1, :ny - 1]
    # fin[5, 0, :] = fout[5, -1, :]  # wrap right
    # fin[5, :, 0] = fout[5, :, -1]  # wrap top
    # vel 7 decreases x and y simultaneously
    fin[7, :nx - 1, :ny - 1] = fout[7, 1:, 1:]
    # fin[7, -1, :] = fout[7, 0, :]  # wrap left
    # fin[7, :, -1] = fout[7, :, 0]  # wrap bottom

    # vel 6 decreases x and increases y
    fin[6, :nx - 1, 1:] = fout[6, 1:, :ny - 1]
    # fin[6, -1, :] = fout[6, 0, :]  # wrap left
    # fin[6, :, 0] = fout[6, :, -1]  # wrap top
    # vel 8 increases x and decreases y
    fin[8, 1:, :ny - 1] = fout[8, :nx - 1, 1:]
    # fin[8, 0, :] = fout[8, -1, :]  # wrap right
    # fin[8, :, -1] = fout[8, :, 0]  # wrap bottom

    fin[0, :, :] = fout[0, :, :]  # vel 0 is stationary (dont act like you didn't forget this for 2 hours)


def step(i):
    global fin_ox, fin_red, fout_ox, fout_red, rho_ox, rho_red
    # Perform one LBM step
    # Outlet BC
    # Equiv. to neumann BC on concentration (null flux)
    fin_ox[left_col, -1, :] = fin_ox[left_col, -2, :]
    fin_red[left_col, -1, :] = fin_red[left_col, -2, :]

    macroscopic()

    if i > delay:
        # Calculate generated current
        # Charge
        q = F * (rho_red[electrode] * concentration_ph) * (dx ** 2) * cell_depth_ph
        # Current
        j_log[i - delay] = q

        # Electrode BC
        rho_ox[electrode] = 1
        rho_red[electrode] = 0
    equilibrium()

    # Zhou He BC
    fin_ox[right_col, 0, :] = feq_ox[right_col, 0, :] + fin_ox[left_col, 0, :] - feq_ox[left_col, 0, :]
    fin_red[right_col, 0, :] = feq_red[right_col, 0, :] + fin_red[left_col, 0, :] - feq_red[left_col, 0, :]

    # BGK collision
    # Collision
    f_plus = .5 * (fin_ox + fin_ox[c_op])
    f_minus = .5 * (fin_ox - fin_ox[c_op])

    feq_plus = .5 * (feq_ox + feq_ox[c_op])
    feq_minus = .5 * (feq_ox - feq_ox[c_op])

    fout_ox = fin_ox - w_o * (f_plus - feq_plus) - w_e * (f_minus - feq_minus)

    g_plus = .5 * (fin_red + fin_red[c_op])
    g_minus = .5 * (fin_red - fin_red[c_op])

    geq_plus = .5 * (feq_red + feq_red[c_op])
    geq_minus = .5 * (feq_red - feq_red[c_op])

    fout_red = fin_red - w_o * (g_plus - geq_plus) - w_e * (g_minus - geq_minus)

    # Bounce-back
    fout_ox[:, obstacle] = fin_ox[c_op][:, obstacle]
    fout_red[:, obstacle] = fin_red[c_op][:, obstacle]

    # Streaming
    stream(fin_ox, fout_ox)
    stream(fin_red, fout_red)


def run(iterations: int, save_to_disk: bool = True, interval: int = 100, continue_last: bool = False):
    # Launches LBM simulation and a parallel thread for saving data to disk
    global rho_ox, rho_red, fin_ox, fin_red, fout_ox, fout_red, j_log, dt

    print(f"Simulating {iterations * dt} seconds")

    j_log = torch.zeros((iterations, electrode.sum()), dtype=torch.float64, device=device)

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
            step(i)  # Perform one LBM step
            if i % interval == 0:
                # Calculate MLUPS by dividing number of nodes by time in seconds
                delta_t = time.time() - start
                mlups = nx * ny * counter / (delta_t * 1e6)
                if save_to_disk:
                    # push data to queue
                    q.put((rho_red.detach().clone(), f"output/{i // interval:05}.png"))  # Five digit filename
                # Reset timer and counter
                start = time.time()
                counter = 0

            counter += 1
            bar.text(f"MLUPS: {mlups:.2f} | Electrode density {rho_ox[electrode].mean().cpu().numpy():.5f}")
            bar()

    # Save final data to numpy files
    np.save(f"output/Electrochemical_last_rho_ox.npy", rho_ox.cpu().numpy())
    np.save(f"output/Electrochemical_last_rho_red.npy", rho_red.cpu().numpy())

    # Plot current density
    plt.show()
    fig, ax = plt.subplots()
    x = np.linspace(1e-1, electrode.sum().cpu().numpy(), 1000)
    lev = 0.67 * (d_ph**(2/3)) * (u_l/(ny*x))**(1/3)

    plt.plot(j_log[iterations - delay - 1].cpu().numpy(), 'gs')
    plt.plot(x, lev, 'r--')
    ax.set_ylabel("Flux")
    # plt.ylim([0, torch.max(j_log[iterations - delay - 1]).cpu().numpy() * 1.1])
    # ax.legend()
    # plt.savefig("output/cottrell_current.png", bbox_inches='tight', pad_inches=0, dpi=900)
    plt.show()
    # fig, ax = plt.subplots()
    # rel_err = j_log[:-1].cpu().numpy() / cott
    # ax.plot(ph_time, rel_err, '-g')
    # ax.set_xlabel("Time (s)")
    # ax.set_ylabel("$I / I_{Cottrell}$")
    # ax.grid()
    # ax.set_ylim(0.8, 2.5)
    # plt.show()

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
        plt.close()


if __name__ == '__main__':
    print(f"omega: {omega_l}")
    run(total_iterations, save_to_disk=True, interval=1000, continue_last=False)
