import matplotlib.pyplot as plt
import torch

from util import *
import threading
import queue
from alive_progress import alive_bar
import time
import numpy as np

# Physical dimensions
cell_length_ph = 5e-2  # 2.5cm or 0.025m
channel_width_ph = 5e-3  # 5mm or 0.005m
depth_ph = 5e-3  # 5mm
diff_ph = 0.76e-9  # m^2/s (From Allen , Bard appendix for Ferrocyanide, page 831)
vel_ph = 0.01  # m/s
Pe = vel_ph * channel_width_ph / diff_ph  # Peclet number
concentration_ph = 100  # mol/m^3 ~ 0.1M

input_image = "input/mmrfbs/MMRFB_v1.png"

obstacle = generate_obstacle_tensor(input_image)
electrode1 = generate_electrode_tensor(input_image, BLUE)
# TODO: properly treat velocity (should be below 0.1)
v_field = torch.from_numpy(np.load("input/mmrfb_u.npy"))
v_field = v_field.to(device)

# Electrode lengths for current densities
electrode1_size = torch.sum(electrode1)

# inlets have different color coding to electrodes
inlet_bottom = generate_electrode_tensor(input_image, GREEN)
inlet_top = generate_electrode_tensor(input_image, YELLOW)


# Diffusion constant
nx, ny = obstacle.shape
omega_l = 1.995
re, dx, dt, ulb = convert_from_physical_params_ns(cell_length_ph, channel_width_ph, vel_ph, diff_ph, nx, omega_l)
input("Press enter to continue...")
v_field = v_field / torch.max(v_field) * ulb

# Initialize
rho_ox_1 = torch.ones((nx, ny), dtype=torch.float64, device=device)
feq_ox_1 = torch.ones((9, nx, ny), device=device)
rho_ox_1[:, :ny // 2] = 0


def equilibrium():
    global feq_ox_1
    cu = torch.einsum('ixy,ji->jxy', v_field, c)
    feq_ox_1 = w.view(9, 1, 1) * rho_ox_1 * (1 + 3 * cu)


equilibrium()

fin_ox_1 = feq_ox_1.clone()
fout_ox_1 = feq_ox_1.clone()


def macroscopic():
    global rho_ox_1
    # With source term correction as shown in Kruger 310
    rho_ox_1 = torch.clamp(torch.sum(fin_ox_1, 0), 0, 10)


def stream(fin, fout):
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

    fin[0, :, :] = fout[0, :, :]


def step(i):
    global fin_ox_1, fout_ox_1

    fin_ox_1[left_col, -1, :] = fin_ox_1[left_col, -2, :]

    macroscopic()
    # Inlet concentrations
    rho_ox_1[inlet_bottom] = 1
    rho_ox_1[inlet_top] = 0
    rho_ox_1[electrode1] = 0

    equilibrium()

    # Zhou He BC
    fin_ox_1[right_col, 0, :] = feq_ox_1[right_col, 0, :] + fin_ox_1[left_col, 0, :] - feq_ox_1[left_col, 0, :]

    fout_ox_1 = fin_ox_1 - omega_l * (fin_ox_1 - feq_ox_1)

    # Bounce Back
    fout_ox_1[:, obstacle] = fin_ox_1[c_op][:, obstacle]
    # Streaming
    stream(fin_ox_1, fout_ox_1)


def run(iterations: int, save_to_disk: bool = True, interval: int = 100, continue_last: bool = False):
    global rho_ox_1, fin_ox_1

    print(f"Simulating {iterations * dt} seconds")

    if continue_last:
        rho_ox_1 = torch.from_numpy(np.load("output/Depletion_last_rho_ox_1.npy")).to(device)

        equilibrium()

        fin_ox_1 = feq_ox_1.clone()

    if save_to_disk:
        q = queue.Queue()
        t = threading.Thread(target=save_data, args=(q,))
        t.start()

    with alive_bar(iterations, force_tty=True) as bar:
        start = time.time()
        counter = 0
        for i in range(iterations):
            step(i)
            if i % interval == 0:
                delta_t = time.time() - start
                mlups = nx * ny * counter / (delta_t * 1e6)
                if save_to_disk:
                    q.put((rho_ox_1.clone(), f"output/{i // interval:05}.png"))
                start = time.time()
                counter = 0
            if i % 1000 == 0:
                # periodically save the state
                np.save(f"output/temp/Depletion_last_rho_ox_1.npy", rho_ox_1.cpu().numpy())
            counter += 1
            bar.text(
                f"MLUPS: {mlups:.2f}")
            bar()

    # save final state
    np.save(f"output/Depletion_last_rho_ox_1.npy", rho_ox_1.cpu().numpy())

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
        plt.imshow(data.cpu().numpy().transpose(), cmap=cmap, vmin=0, vmax=1.5)
        plt.colorbar()
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=500)
        plt.close()


if __name__ == "__main__":
    run(5000, save_to_disk=True, interval=1000, continue_last=False)
