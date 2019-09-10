from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator as sp
from amp_simple_nn.convert import make_amp_descriptors_simple_nn
import numpy as np

atoms = molecule('O2')
atoms.set_cell([10,10,10])
print(atoms.positions)

atoms.set_calculator(sp(atoms=atoms, energy = -1,
                        forces = np.array([[-1,-1,-1],[-1,-1,-1]])))

images = [atoms]
g2_etas = [0.005]
g2_rs_s = [0] * 4
g4_etas = [0.005]
g4_zetas = [1., 4.]
g4_gammas = [1., -1.]
cutoff = 4
make_amp_descriptors_simple_nn(images,g2_etas,g2_rs_s,g4_etas,
                               g4_zetas,g4_gammas,cutoff)

