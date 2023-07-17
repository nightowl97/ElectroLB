import matplotlib.pyplot as plt
import torch

from util import *
import threading
import queue
from alive_progress import alive_bar
import time
import numpy as np

# Physical dimensions
length_ph = 5e-2  # 5mm or 0.005m
height_ph = 5e-3  # 2mm or 0.02m
diff_ph = 0.76e-8  # m^2/s (From Allen, Bard appendix for Ferrocyanide, page 831)
vel_ph = 0.01  # m/s
Pe = vel_ph * height_ph / diff_ph  # Peclet number

# External circuit load
resistor = torch.ones(10, device=device)  # ohm
voltage_drop = 0  # volts
j = 0  # current global variable
# Formal potentials
E_01 = 0.6
E_02 = -0.6

input_image = "input/mmrfbs/planar.png"

obstacle = generate_obstacle_tensor(input_image)
electrode1 = generate_electrode_tensor(input_image, BLUE)
electrode2 = generate_electrode_tensor(input_image, RED)
v_field = 300 * torch.from_numpy(np.load("output/BaseLattice_last_u.npy"))
v_field = v_field.clone().to(device)

# Electrode lengths for current densities
electrode1_size = electrode1.size(0)
electrode2_size = electrode2.size(0)

# inlets have different color coding to electrodes
inlet_bottom = generate_electrode_tensor(input_image, GREEN)
inlet_top = generate_electrode_tensor(input_image, YELLOW)


# Diffusion constant
d = obstacle.shape[0] * 0.3 / Pe  # Lowering omega requires increasing the resolution
tau = 3 * d + .5
omega = 1 / tau
nx, ny = obstacle.shape

# Initialize
rho_ox_1 = torch.zeros((nx, ny), dtype=torch.float64, device=device)
rho_red_1 = torch.zeros((nx, ny), dtype=torch.float64, device=device)
rho_ox_2 = torch.zeros((nx, ny), dtype=torch.float64, device=device)
rho_red_2 = torch.zeros((nx, ny), dtype=torch.float64, device=device)

feq_ox_1 = torch.zeros((9, nx, ny), device=device)
feq_red_1 = torch.zeros((9, nx, ny), device=device)
feq_ox_2 = torch.zeros((9, nx, ny), device=device)
feq_red_2 = torch.zeros((9, nx, ny), device=device)


def equilibrium():
    global feq_ox_1, feq_red_1, feq_ox_2, feq_red_2
    cu = torch.einsum('ixy,ji->jxy', v_field, c)
    feq_ox_1 = w.view(9, 1, 1) * rho_ox_1 * (1 + 3 * cu)
    feq_red_1 = w.view(9, 1, 1) * rho_red_1 * (1 + 3 * cu)
    feq_ox_2 = w.view(9, 1, 1) * rho_ox_2 * (1 + 3 * cu)
    feq_red_2 = w.view(9, 1, 1) * rho_red_2 * (1 + 3 * cu)


equilibrium()

fin_ox_1 = feq_ox_1.clone()
fin_red_1 = feq_red_1.clone()
fin_ox_2 = feq_ox_2.clone()
fin_red_2 = feq_red_2.clone()

fout_ox_1 = feq_ox_1.clone()
fout_red_1 = feq_red_1.clone()
fout_ox_2 = feq_ox_2.clone()
fout_red_2 = feq_red_2.clone()

source_ox_1 = torch.zeros_like(rho_ox_1, device=device)
source_red_1 = torch.zeros_like(rho_red_1, device=device)
source_ox_2 = torch.zeros_like(rho_ox_2, device=device)
source_red_2 = torch.zeros_like(rho_red_2, device=device)

# population source terms
source_ox_1_i = torch.einsum('i,jk->ijk', w, source_ox_1)
source_red_1_i = torch.einsum('i,jk->ijk', w, source_red_1)
source_ox_2_i = torch.einsum('i,jk->ijk', w, source_ox_2)
source_red_2_i = torch.einsum('i,jk->ijk', w, source_red_2)

# Set up Nernst potential over interface
e_nernst_1 = E_01 - (R * T / F) * torch.log(rho_red_1[electrode1] / rho_ox_1[electrode1])
e_nernst_2 = E_02 - (R * T / F) * torch.log(rho_red_2[electrode2] / rho_ox_2[electrode2])


def macroscopic():
    global rho_ox_1, rho_red_1, rho_ox_2, rho_red_2
    # With source term correction as shown in Kruger 310
    rho_ox_1 = torch.clamp(torch.sum(fin_ox_1 + source_ox_1_i / 2, 0), 0, 10)
    rho_red_1 = torch.clamp(torch.sum(fin_red_1 + source_red_1_i / 2, 0), 0, 10)
    rho_ox_2 = torch.clamp(torch.sum(fin_ox_2 + source_ox_2_i / 2, 0), 0, 10)
    rho_red_2 = torch.clamp(torch.sum(fin_red_2 + source_red_2_i / 2, 0), 0, 10)


def stream(fin, fout):
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

    fin[0, :, :] = fout[0, :, :]


def step(i):
    global fin_ox_1, fin_ox_2, fin_red_1, fin_red_2, e_nernst_1, voltage_drop, source_ox_1_i, source_red_1_i
    global fout_ox_1, fout_ox_2, fout_red_1, fout_red_2, e_nernst_2, j, source_ox_2_i, source_red_2_i

    fin_ox_1[left_col, -1, :] = fin_ox_1[left_col, -2, :]
    fin_red_1[left_col, -1, :] = fin_red_1[left_col, -2, :]
    fin_ox_2[left_col, -1, :] = fin_ox_2[left_col, -2, :]
    fin_red_2[left_col, -1, :] = fin_red_2[left_col, -2, :]

    macroscopic()
    # Inlet concentrations
    rho_ox_1[inlet_bottom] = 1
    rho_red_1[inlet_bottom] = 1
    rho_ox_1[inlet_top] = 0
    rho_red_1[inlet_top] = 0
    rho_ox_2[inlet_top] = 1
    rho_red_2[inlet_top] = 1
    rho_ox_2[inlet_bottom] = 0
    rho_red_2[inlet_bottom] = 0

    equilibrium()

    # Zhou He BC
    fin_ox_1[right_col, 0, :] = feq_ox_1[right_col, 0, :] + fin_ox_1[left_col, 0, :] - feq_ox_1[left_col, 0, :]
    fin_red_1[right_col, 0, :] = feq_red_1[right_col, 0, :] + fin_red_1[left_col, 0, :] - feq_red_1[left_col, 0, :]
    fin_ox_2[right_col, 0, :] = feq_ox_2[right_col, 0, :] + fin_ox_2[left_col, 0, :] - feq_ox_2[left_col, 0, :]
    fin_red_2[right_col, 0, :] = feq_red_2[right_col, 0, :] + fin_red_2[left_col, 0, :] - feq_red_2[left_col, 0, :]

    # Average potential difference between two electrodes
    # e_nernst_1 = E_01 - (R * T / F) * torch.log(rho_red_1[electrode1] / rho_ox_1[electrode1])
    # e_nernst_2 = E_02 - (R * T / F) * torch.log(rho_red_2[electrode2] / rho_ox_2[electrode2])

    # Resulting average current
    # voltage_drop = torch.mean(e_nernst_1) - torch.mean(e_nernst_2)
    # J_ohm = voltage_drop / resistor[i]
    # q_diff_1 = rho_red_1[electrode1].sum(0)
    # q_diff_2 = rho_ox_2[electrode2].sum(0)
    # q_diff = torch.min(q_diff_1, q_diff_2)
    # # If resistance limited
    # if J_ohm < q_diff:
    #     q_1 = J_ohm / (2 * electrode1_size)
    #     q_2 = J_ohm / (2 * electrode2_size)
    #
    #     j = J_ohm
    #
    #     # Source Terms
    #     source_ox_1[electrode1] = q_1
    #     source_red_1[electrode1] = -q_1
    #
    #     source_ox_2[electrode2] = -q_2
    #     source_red_2[electrode2] = q_2
    # else:  # If diffusion limited
    #     # locate depletion regions in electrodes
    #     mask_1 = q_diff / electrode1_size > rho_red_1[electrode1] / F  # Where we try to draw more than available
    #     mask_2 = q_diff / electrode2_size > rho_ox_2[electrode2] / F  # Where we try to draw more than available
    #
    #     source_ox_1[electrode1][mask_1] = rho_red_1[electrode1][mask_1]
    #     source_red_1[electrode1][mask_1] = -rho_red_1[electrode1][mask_1]
    #     j_initial = torch.sum(rho_red_1[electrode1][mask_1])
    #     rem_j = q_diff - j_initial  # remaining current to be drawn from the other sites
    #     source_ox_1[electrode1][~mask_1] = rem_j / electrode1_size
    #
    #     source_ox_2[electrode2][mask_2] = -rho_ox_2[electrode2][mask_2]
    #     source_red_2[electrode2][mask_2] = rho_ox_2[electrode2][mask_2]
    #     j_initial = torch.sum(rho_ox_2[electrode2][mask_2])
    #     rem_j = q_diff - j_initial  # remaining current to be drawn from the other sites
    #     source_red_2[electrode2][~mask_2] = rem_j / electrode2_size
    #
    #     j = q_diff
    #     voltage_drop = j * resistor[i]

    # BGK collision with source terms (Kruger page 310)
    source_ox_1_i = torch.einsum('i,jk->ijk', w, source_ox_1)
    source_red_1_i = torch.einsum('i,jk->ijk', w, source_red_1)
    source_ox_2_i = torch.einsum('i,jk->ijk', w, source_ox_2)
    source_red_2_i = torch.einsum('i,jk->ijk', w, source_red_2)
    fout_ox_1 = fin_ox_1 - omega * (fin_ox_1 - feq_ox_1) + (1-1/(2 * tau)) * source_ox_1_i
    fout_red_1 = fin_red_1 - omega * (fin_red_1 - feq_red_1) +(1-1/(2*tau))*source_red_1_i
    fout_ox_2 = fin_ox_2 - omega * (fin_ox_2 - feq_ox_2) + (1-1/(2 * tau)) * source_ox_2_i
    fout_red_2 = fin_red_2 - omega * (fin_red_2 - feq_red_2) +(1-1/(2*tau))*source_red_2_i

    # Bounce Back
    fout_ox_1[:, obstacle] = fin_ox_1[c_op][:, obstacle]
    fout_red_1[:, obstacle] = fin_red_1[c_op][:, obstacle]
    fout_ox_2[:, obstacle] = fin_ox_2[c_op][:, obstacle]
    fout_red_2[:, obstacle] = fin_red_2[c_op][:, obstacle]
    # Streaming
    stream(fin_ox_1, fout_ox_1)
    stream(fin_red_1, fout_red_1)
    stream(fin_ox_2, fout_ox_2)
    stream(fin_red_2, fout_red_2)


def run(iterations: int, save_to_disk: bool = True, interval: int = 100, continue_last: bool = False):
    global rho_ox_1, rho_red_1, fin_ox_1, fin_red_1, fout_ox_1, fout_red_1, voltage_drop
    global rho_ox_2, rho_red_2, fin_ox_2, fin_red_2, fout_ox_2, fout_red_2, j, resistor

    # Track voltage and current
    v_track = np.zeros(iterations)
    j_track = np.zeros(iterations)
    resistor = 1 - 0.5 * torch.heaviside(torch.arange(iterations) - torch.tensor(iterations) // 10, torch.tensor(1))
    plt.plot(resistor)
    plt.show()

    if continue_last:
        rho_ox_1 = torch.from_numpy(np.load("output/Electrochemical_last_rho_ox_1.npy")).to(device)
        rho_red_1 = torch.from_numpy(np.load("output/Electrochemical_last_rho_red_1.npy")).to(device)
        rho_ox_2 = torch.from_numpy(np.load("output/Electrochemical_last_rho_ox_2.npy")).to(device)
        rho_red_2 = torch.from_numpy(np.load("output/Electrochemical_last_rho_red_2.npy")).to(device)

        equilibrium()

        fin_ox_1 = feq_ox_1.clone()
        fin_red_1 = feq_red_1.clone()
        fin_ox_2 = feq_ox_2.clone()
        fin_red_2 = feq_red_2.clone()
        fout_ox_1 = feq_ox_1.clone()
        fout_red_1 = feq_red_1.clone()
        fout_ox_2 = feq_ox_2.clone()
        fout_red_2 = feq_red_2.clone()

    if save_to_disk:
        q = queue.Queue()
        t = threading.Thread(target=save_data, args=(q,))
        t.start()

    with alive_bar(iterations) as bar:
        start = time.time()
        counter = 0
        for i in range(iterations):
            step(i)
            if i % interval == 0:
                dt = time.time() - start
                mlups = nx * ny * counter / (dt * 1e6)
                if save_to_disk:
                    q.put((rho_ox_1.clone(), f"output/{i // interval:05}.png"))
                start = time.time()
                counter = 0
            counter += 1
            # Log current and voltage
            v_track[i] = voltage_drop
            j_track[i] = j
            bar.text(f"MLUPS: {mlups:.2f} | current {j:.5f} | voltage {voltage_drop:.5f}")
            bar()

    # save final state
    np.save(f"output/Electrochemical_last_rho_ox_1.npy", rho_ox_1.cpu().numpy())
    np.save(f"output/Electrochemical_last_rho_red_1.npy", rho_red_1.cpu().numpy())
    np.save(f"output/Electrochemical_last_rho_ox_2.npy", rho_ox_2.cpu().numpy())
    np.save(f"output/Electrochemical_last_rho_red_2.npy", rho_red_2.cpu().numpy())

    print(f"Last I and V are:\t{j_track[-1]}, and {v_track[-1]}")
    fig, ax = plt.subplots()
    ax.plot(v_track)
    plt.show()
    plt.close()

    if save_to_disk:
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


if __name__ == "__main__":
    print(f"omega: {omega}")
    run(10000, save_to_disk=True, interval=100, continue_last=True)
