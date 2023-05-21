import queue
import threading
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from alive_progress import alive_bar
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cmap = plt.get_cmap('Purples')
cmap.set_bad((1, 0, 0, 1))


# generate tensor from image with black pixels as obstacles
def generate_obstacle_tensor(image_path):
    image = Image.open(image_path)
    image = np.asarray(image.convert('L'))

    # black pixels are True, white pixels are False
    image = torch.tensor(image == 0, dtype=torch.bool).T.to(device)
    return image


def plot_data(q):
    # This function will run in a separate thread and plot the data in the queue
    while True:
        data, filename = q.get()
        if data is None:
            break
        plt.clf()
        plt.axis("off")
        mod_u = torch.sqrt(data[0]**2 + data[1]**2)
        mod_u[obstacle] = np.nan
        plt.imshow(mod_u.cpu().numpy().transpose(), cmap=cmap)
        plt.savefig(filename)


save_disk = True
obstacle = generate_obstacle_tensor('input/inputbig.png')
maxiter = 10000
Re = 2000.
# dimensions from obstacle tensor
nx, ny = obstacle.shape
ly = ny - 1
uLB = 0.04
nulB = uLB * ny / Re
omega = 1 / (3 * nulB + 0.5)

v = torch.tensor([[1, 1], [1, 0], [1, -1], [0, 1], [0, 0], [0, -1], [-1, 1], [-1, 0], [-1, -1]], device=device).float()
t = torch.tensor([1 / 36, 1 / 9, 1 / 36, 1 / 9, 4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36], device=device)
col1 = torch.tensor([0, 1, 2], device=device)
col2 = torch.tensor([3, 4, 5], device=device)
col3 = torch.tensor([6, 7, 8], device=device)


def macroscopic(fin):
    rho = fin.sum(0)
    u = torch.einsum('ji,jkl->ikl', v, fin) / rho
    return rho, u


def equilibrium(rho, u):
    usqr = 3/2 * (u[0] ** 2 + u[1] ** 2)
    feq = torch.zeros((9, nx, ny), device=device)
    cu = 3 * torch.einsum('ijk,li->ljk', u, v)
    feq = rho * t.view(9, 1, 1) * (1 + cu + 0.5 * cu ** 2 - usqr)
    return feq


vel = 0.04 * torch.ones((ny), device=device).float()
fin = equilibrium(torch.ones((nx, ny), device=device).float(), torch.zeros((2, nx, ny), device=device).float())
rho, u = macroscopic(fin)
fout = fin.clone()
feq = fin.clone()

# Create a queue and a thread to handle the plotting
plot_queue = queue.Queue()
plot_thread = threading.Thread(target=plot_data, args=(plot_queue,))
plot_thread.start()

with alive_bar(maxiter) as bar:
    prev_time = time.time()
    counter = 0
    for it in range(maxiter):
        # Outlet Boundary Condition
        fin[col3, -1, :] = fin[col3, -2, :]

        # Momenta
        rho, u = macroscopic(fin)

        u[0, 0, :] = vel
        rho[0, :] = 1 / (1 - u[0, 0, :]) * (torch.sum(fin[col2, 0, :], dim=0) + 2 * torch.sum(fin[col3, 0, :], dim=0))

        # Equilibrium
        feq = equilibrium(rho, u)

        # Inlet Boundary Condition
        fin[[0, 1, 2], 0, :] = feq[[0, 1, 2], 0, :] + fin[[8, 7, 6], 0, :] - feq[[8, 7, 6], 0, :]

        # BGK collision
        fout = fin - omega * (fin - feq)

        # BounceBack
        fin_flipped = torch.flip(fin, [0])
        fout[:, obstacle] = fin_flipped[:, obstacle]

        # Streaming
        for i in range(9):
            fin[i, :, :] = torch.roll(fout[i, :, :], shifts=int(v[i, 0].item()), dims=0)
            fin[i, :, :] = torch.roll(fin[i, :, :], shifts=int(v[i, 1].item()), dims=1)

        if it % 100 == 0:
            if save_disk:
                plot_queue.put((u.clone().cpu(), "output/gpu_vel{0:03d}.png".format(it // 100)))
            # show performance
            current_time = time.time()
            dt = current_time - prev_time
            mlups = nx * ny * counter / (dt * 1e6)
            prev_time = time.time()
            counter = 0

        counter += 1
        bar.text("MLUPS: {0:.2f}".format(mlups))

        bar()

# When all computations are done, signal the plot thread to finish
plot_queue.put((None, None))
plot_thread.join()
