"""
We calculate the (6x6) friction tensor for 
full-coverage CO adsorbed on a Cu(100) top site.
"""

import numpy as np
from ase.all import *
import os, sys
import coolvib
from scipy import linalg as LA

#####DEFINE SYSTEM and PARAMETERS######
system = read('CO-on-Cu100/eq/geometry.in')
cell = system
active_atoms = [3,4] # meaning we have two atoms - C - 0 and O - 1

model = coolvib.workflow_tensor(system, code='aims', active_atoms=active_atoms)

finite_difference_incr = 0.001

keywords = {
    'discretization_type' : 'gaussian',
    'discretization_broadening' : 0.05,
    'discretization_length' : 0.01,
    'max_energy' : 6.00,
    'temperature' : 300,
    'delta_function_type': 'gaussian',
    'delta_function_width': 0.80,
    'perturbing_energy' : 0.0,
        }

print 'workflow initialized and keywords set'

######READ QM INPUT DATA###
model.read_input_data(
        spin=True, 
        path='./CO-on-Cu100',
        filename='OUTPUT', 
        active_atoms=active_atoms, 
        incr=finite_difference_incr,
        )
print 'successfully read QM input data'

######CALCULATE SPECTRAL FUNCTION###
model.calculate_spectral_function(mode='default', **keywords)
print 'successfully calculated spectral_function'
model.print_spectral_function('nacs-spectrum.out')

######CALCULATE FRICTION TENSOR###
model.calculate_friction_tensor(**keywords)
print 'successfully calculated friction tensor'

model.analyse_friction_tensor()

model.print_jmol_friction_eigenvectors()
