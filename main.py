from ElectroLB import *

Re = 6000.
obstacle = generate_obstacle_tensor('inputbig.png')
lattice = BaseLattice(obstacle, Re)
lattice.run(100000)
