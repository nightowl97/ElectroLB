import torch

from ElectroLB import *
import numpy as np

Re = 20.
obstacle = generate_obstacle_tensor('input/halfcell.png')
lattice = BaseLattice(obstacle, Re)
lattice.run(50000, save_data=True)

# Load initial rho and u from file
# init_rho = np.load('output/BaseLattice_last_rho.npy')
# init_u = np.load('output/BaseLattice_last_u.npy')
# lattice2 = BaseLattice(obstacle, Re, initial_rho=init_rho, initial_u=init_u)
# lattice2.run(1000, save_data=True)

# Electrolattice
# v_field = torch.from_numpy(np.load('output/BaseLattice_last_u.npy'))
# v_field = torch.zeros((2, 400, 400))
# v_field[0, :, :] = 0.04
# electrolattice = ElectroLattice(v_field, obstacle, 0.05)
# electrolattice.run(50000, save_data=True)
