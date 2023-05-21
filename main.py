from ElectroLB import *
import numpy as np

Re = 6000.
obstacle = generate_obstacle_tensor('input/inputbig.png')
lattice = BaseLattice(obstacle, Re)
# lattice.run(4000, save_data=False)

# Load initial rho and u from file
init_rho = np.load('output/last_rho.npy')
init_u = np.load('output/last_u.npy')
lattice2 = BaseLattice(obstacle, Re, initial_rho=init_rho, initial_u=init_u)
lattice2.run(6000)