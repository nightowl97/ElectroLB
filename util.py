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
YELLOW = np.asarray([255, 255, 0])
CYAN = np.asarray([0, 255, 255])
MAGENTA = np.asarray([255, 0, 255])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

cmap = plt.get_cmap('coolwarm')
cmap.set_bad((0, 0, 0, 1))

# Physical constants
T = 298  # Kelvin
R = 8.3145  # J/mol.K
F = 96485  # C/mol

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


def generate_electrode_tensor(file, color=BLUE):
    # Generate electrode tensor from image file
    img_array = np.asarray(Image.open(file))
    # Black pixels are True, white pixels are False

    electrode = (img_array == color).all(axis=2).T
    electrode = torch.tensor(electrode, dtype=torch.bool).to(device)
    return electrode


def save_data(q: queue.Queue, obstacle):
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


def convert_to_physical_velocity(velocity_array, dx, dt):
    # Convert lattice velocity to physical velocity
    # C_factor * lattice_velocity = physical_velocity
    conversion_factor = dx / dt  # lattice velocity to physical velocity
    return velocity_array * conversion_factor


def convert_to_lattice_velocity(velocity_array, dx, dt):
    return velocity_array / (dx / dt)


def get_lattice_viscosity_from_tau_l(tau):
    """Kruger page 272"""
    return (1 / 3) * (tau - .5)


def convert_from_physical_params_ns(total_length_ph, channel_width_ph, char_velocity_ph, viscosity_ph,
                                    lattice_size, omega_l):
    # Kruger page 283 example
    re = channel_width_ph * char_velocity_ph / viscosity_ph
    dx = total_length_ph / lattice_size
    inlet_width_l = channel_width_ph / dx

    tau_l = 1 / omega_l
    nu_l = get_lattice_viscosity_from_tau_l(tau_l)

    dt = (1 / 3) * ((tau_l - 0.5) * dx ** 2) / viscosity_ph
    ulb = char_velocity_ph / (dx / dt)  # C_factor * lattice_velocity = physical_velocity
    assert np.abs(ulb - (re * nu_l / inlet_width_l)) < 1e-5
    print("Simulation parameters (Navier-Stokes):")
    print(f"Re: {re}")
    print(f"dt: {dt}")
    print(f"dx: {dx}")
    print(f"ulb: {ulb}")
    print(f"omega_l: {omega_l}")
    print(f"nu_l: {nu_l}")
    return re, dx, dt, ulb


def convert_from_physical_params_diff(total_length_ph, channel_width_ph, char_velocity_ph, diffusion_coeff_ph,
                                      lattice_size, omega_l):
    pe = char_velocity_ph * channel_width_ph / diffusion_coeff_ph
    dx = total_length_ph / lattice_size
    inlet_width_l = channel_width_ph / dx

    tau_l = 1 / omega_l
    dt = (1 / 3) * ((tau_l - 0.5) * dx ** 2) / diffusion_coeff_ph
    u_l = char_velocity_ph / (dx / dt)
    d_l = get_lattice_viscosity_from_tau_l(tau_l)

    print("Simulation parameters (Diffusion):")
    print(f"Pe: {pe}")
    print(f"dt: {dt}")
    print(f"dx: {dx}")
    print(f"u_l: {u_l}")
    print(f"omega_l: {omega_l}")
    print(f"d_l: {d_l}")
    return pe, dx, dt, d_l


def convert_from_physical_params_pure_diff(total_length_ph, diffusion_coeff_ph, lattice_size, omega_l,
                                           total_iterations):
    dx = total_length_ph / lattice_size

    tau_l = 1 / omega_l
    dt = (1 / 3) * ((tau_l - 0.5) * dx ** 2) / diffusion_coeff_ph
    d_l = get_lattice_viscosity_from_tau_l(tau_l)

    fo = d_l * total_iterations / lattice_size  # Fourier number
    print("Simulation parameters (Diffusion):")
    print(f"Fo: {fo}")
    print(f"dt: {dt}")
    print(f"dx: {dx}")
    print(f"omega_l: {omega_l}")
    print(f"d_l: {d_l}")
    return fo, dx, dt, d_l
