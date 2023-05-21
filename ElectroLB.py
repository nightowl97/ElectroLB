import matplotlib.pyplot as plt
import torch
import threading
import queue
from PIL import Image
import numpy as np
from alive_progress import alive_bar
import time

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cmap = plt.get_cmap('Purples')
cmap.set_bad((1, 0, 0, 1))


class BaseLattice:
    """
    Base class for all lattices that handles the basic logic of LBM for the Navier-Stokes equations
    Outgoing directions:
    6---2---5
    | \ | / |
    3---0---1
    | / | \ |
    7---4---8
    """
    c = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
                     device=device).float()
    c_op = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], device=device)  # Opposite directions indices
    w = torch.tensor([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], device=device)  # weights
    right_col = [1, 5, 8]  # Right column of velocities
    left_col = [3, 7, 6]  # Left column of velocities (order is important, see line 85 equilibrium function in inlet)
    center_col = [0, 2, 4]  # Center column of velocities

    # Base class for all lattices that handles the basic logic of LBM
    def __init__(self, obstacle: torch.tensor, re: float, initial_rho: torch.tensor = None,
                 initial_u: torch.tensor = None):
        # Create obstacle tensor from numpy array
        self.obstacle = obstacle.clone().to(device)
        self.nx, self.ny = obstacle.shape  # Number of nodes in x and y directions
        self.re = re  # Reynolds number
        self.ulb = 0.04  # characteristic velocity
        self.nulb = self.ulb * self.ny / self.re  # kinematic viscosity
        self.omega = 1 / (3 * self.nulb + 0.5)  # relaxation parameter

        # Initialize macroscopic variables
        if initial_rho is not None:
            self.rho = torch.tensor(initial_rho.copy()).to(device)
        else:
            self.rho = torch.ones((self.nx, self.ny), device=device).float()
        if initial_u is not None:
            self.u = torch.tensor(initial_u.copy()).to(device)
        else:
            self.u = torch.zeros((2, self.nx, self.ny), device=device).float()

        # Initialize populations
        self.feq = torch.zeros((9, self.nx, self.ny), device=device).float()
        self.equilibrium()  # Initialize equilibrium populations
        self.fin = self.feq.clone()  # Initialize incoming populations (pre-collision)
        self.fout = self.feq.clone()  # Initialize outgoing populations (post-collision)

    def macroscopic(self):
        # Calculate macroscopic variables rho and u (Kruger et al., page 63)
        self.rho = self.fin.sum(0)  # Sum along first axis (populations in each node)
        self.u = torch.einsum('ji,jxy->ixy', self.c, self.fin) / self.rho

    def equilibrium(self):
        # Calculate equilibrium populations (Kruger et al., page 64)
        usqr = 3 / 2 * (self.u[0] ** 2 + self.u[1] ** 2)
        cu = 3 * torch.einsum('ixy,ji->jxy', self.u, self.c)  # previously ijk,li->ljk
        self.feq = self.rho * self.w.view(9, 1, 1) * (1 + cu + 0.5 * cu ** 2 - usqr)

    def stream(self):
        """
        6---2---5
        | \ | / |
        3---0---1
        | / | \ |
        7---4---8
        """
        # Streaming periodically
        nx, ny = self.nx, self.ny
        self.fin[1, 1:, :] = self.fout[1, :nx - 1, :]  # vel 1 increases x
        self.fin[1, 0, :] = self.fout[1, -1, :]  # wrap
        self.fin[3, :nx - 1, :] = self.fout[3, 1:, :]  # vel 3 decreases x
        self.fin[3, -1, :] = self.fout[3, 0, :]  # wrap

        self.fin[2, :, 1:] = self.fout[2, :, :ny - 1]  # vel 2 increases y
        self.fin[2, :, 0] = self.fout[2, :, -1]  # wrap
        self.fin[4, :, :ny - 1] = self.fout[4, :, 1:]  # vel 4 decreases y
        self.fin[4, :, -1] = self.fout[4, :, 0]  # wrap

        # vel 5 increases x and y simultaneously
        self.fin[5, 1:, 1:] = self.fout[5, :nx - 1, :ny - 1]
        self.fin[5, 0, :] = self.fout[5, -1, :]  # wrap right
        self.fin[5, :, 0] = self.fout[5, :, -1]  # wrap top
        # vel 7 decreases x and y simultaneously
        self.fin[7, :nx - 1, :ny - 1] = self.fout[7, 1:, 1:]
        self.fin[7, -1, :] = self.fout[7, 0, :]  # wrap left
        self.fin[7, :, -1] = self.fout[7, :, 0]  # wrap bottom

        # vel 6 decreases x and increases y
        self.fin[6, :nx - 1, 1:] = self.fout[6, 1:, :ny - 1]
        self.fin[6, -1, :] = self.fout[6, 0, :]  # wrap left
        self.fin[6, :, 0] = self.fout[6, :, -1]  # wrap top
        # vel 8 increases x and decreases y
        self.fin[8, 1:, :ny - 1] = self.fout[8, :nx - 1, 1:]
        self.fin[8, 0, :] = self.fout[8, -1, :]  # wrap right
        self.fin[8, :, -1] = self.fout[8, :, 0]  # wrap bottom

        self.fin[0, :, :] = self.fout[0, :, :]  # vel 0 is stationary (dont act like you didn't forget this for 2 hours)

    def step(self):
        # Perform one LBM step
        # Outlet BC
        # Doing this first is more stable for some reason
        self.fin[self.left_col, -1, :] = self.fin[self.left_col, -2, :]

        self.macroscopic()  # Calculate macroscopic variables
        # Impose conditions on macroscopic variables
        self.u[0, 0, :] = self.ulb * torch.ones(self.ny, device=device).float()
        self.rho[0, :] = 1 / (1 - self.u[0, 0, :]) * (torch.sum(self.fin[self.center_col, 0, :], dim=0) +
                                                      2 * torch.sum(self.fin[self.left_col, 0, :], dim=0))

        # Equilibrium
        self.equilibrium()

        # Boundary conditions on populations
        # Zou-He BC Fin = Feq + Fin(op) - Feq(op)
        self.fin[self.right_col, 0, :] = self.feq[self.right_col, 0, :] + self.fin[self.left_col, 0, :] \
                                         - self.feq[self.left_col, 0, :]

        # BGK collision
        self.fout = self.fin - self.omega * (self.fin - self.feq)

        # Bounce-back
        self.fout[:, self.obstacle] = self.fin[self.c_op][:, self.obstacle]

        # Streaming
        self.stream()

    def run(self, iterations: int, save_data: bool = True, interval: int = 100):
        # Launches LBM simulation and a parallel thread for saving data to disk

        if save_data:
            # Create queue for saving data to disk
            q = queue.Queue()
            # Create thread for saving data
            t = threading.Thread(target=self.save_data, args=(q,))
            t.start()

        # Run LBM for specified number of iterations
        with alive_bar(iterations) as bar:
            start = time.time()
            counter = 0
            for i in range(iterations):
                self.step()  # Perform one LBM step
                if i % interval == 0:
                    # Calculate MLUPS by dividing number of nodes by time in seconds
                    dt = time.time() - start
                    mlups = self.nx * self.ny * counter / (dt * 1e6)
                    if save_data:
                        # push data to queue
                        q.put((self.u, f"output/{i // interval:05}.png"))
                    # Reset timer and counter
                    start = time.time()
                    counter = 0

                counter += 1
                bar.text(f"MLUPS: {mlups:.2f}")
                bar()

        # Save final data to numpy files
        np.save("output/last_u.npy", self.u.cpu().numpy())
        np.save("output/last_rho.npy", self.rho.cpu().numpy())

        if save_data:
            # Stop thread for saving data
            q.put((None, None))
            t.join()

    def save_data(self, q: queue.Queue):
        # Save data to disk by running a separate thread that gets data from a queue
        while True:
            data, filename = q.get()
            if data is None:
                break
            plt.clf()
            plt.axis('off')
            usqr = data[0] ** 2 + data[1] ** 2
            usqr[self.obstacle] = np.nan
            plt.imshow(np.sqrt(usqr.cpu().numpy().transpose()), cmap=cmap)
            plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=500)


def generate_obstacle_tensor(file):
    # Generate obstacle tensor from image file
    img_array = np.asarray(Image.open(file).convert('L'))
    # Black pixels are True, white pixels are False
    obstacle = torch.tensor(img_array == 0, dtype=torch.bool).T.to(device)
    return obstacle
