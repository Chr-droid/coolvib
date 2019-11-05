#    This file is part of coolvib
#
#        coolvib is free software: you can redistribute it and/or modify
#        it under the terms of the GNU General Public License as published by
#        the Free Software Foundation, either version 3 of the License, or
#        (at your option) any later version.
#
#        coolvib is distributed in the hope that it will be useful,
#        but WITHOUT ANY WARRANTY; without even the implied warranty of
#        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#        GNU General Public License for more details.
#
#        You should have received a copy of the GNU General Public License
#        along with coolvib.  If not, see <http://www.gnu.org/licenses/>.
"""
restart_test.py

This program takes care of the construction of the coefficient matrices

"""

import numpy as np
from aims import *
from ase.io import *
from scipy.io import FortranFile
from write_restart import *

atoms = read('/home/christian/vsc/energy_decomposition/restart_routines/test_systems/H2_minimal/spin_collinear/geometry.in')
cell = atoms.cell

filename = '/home/christian/vsc/energy_decomposition/restart_routines/test_systems/H2_minimal/spin_collinear/output.aimsrestart'

fermi_level, kpoint_weights = aims_read_fermi_and_kpoints(filename, cell)

nkpts = len(kpoint_weights)

eigenvaluesAB, psiAB, occ_AB, orb_posAB = aims_read_eigenvalues_and_coefficients(fermi_level, '/home/christian/vsc/energy_decomposition/restart_routines/test_systems/H2_minimal/spin_collinear', spin=False, debug=False)
eigenvaluesA, psiA, occA, orb_posA = aims_read_eigenvalues_and_coefficients(fermi_level, '/home/christian/vsc/energy_decomposition/restart_routines/test_systems/H2_minimal/spin_collinear/fragA', spin=False, debug=False)
eigenvaluesB, psiB, occB, orb_posB = aims_read_eigenvalues_and_coefficients(fermi_level, '/home/christian/vsc/energy_decomposition/restart_routines/test_systems/H2_minimal/spin_collinear/fragB', spin=False, debug=False)
#----------- eigenvalues = np.zeros([n_kpts, n_spin, n_states])
#----------- occ = np.zeros([n_kpts, n_spin, n_states])
#----------- psi = np.zeros([n_kpts, n_spin, n_states, n_basis],dtype=complex)
#----------- orbital_pos = np.zeros(n_basis,dtype=np.int)

HAB, SAB = aims_read_HS('/home/christian/vsc/energy_decomposition/restart_routines/test_systems/H2_minimal/spin_collinear',spin=False)
HA, SA = aims_read_HS('/home/christian/vsc/energy_decomposition/restart_routines/test_systems/H2_minimal/spin_collinear/fragA',spin=False)
HB, SB = aims_read_HS('/home/christian/vsc/energy_decomposition/restart_routines/test_systems/H2_minimal/spin_collinear/fragB',spin=False)

nspin = 1 #number-1
n_k = 16    #number-1

#Building the coefficient matrices of the fragments CA, CB and the combined system CAB
CA = np.zeros(shape=(psiA.shape[2],psiA.shape[3]),dtype='complex128')
CB = np.zeros(shape=(psiB.shape[2],psiB.shape[3]),dtype='complex128')

CAB = np.zeros(shape=(n_k,nspin,psiA.shape[2]+psiB.shape[2],psiA.shape[3]+psiB.shape[3]),dtype='complex128')
CCAB = np.zeros(shape=(n_k,nspin,psiA.shape[3]+psiB.shape[3],psiA.shape[2]+psiB.shape[2]),dtype='complex128')
occAB= np.zeros(shape=(n_k,nspin,psiA.shape[2]+psiB.shape[2]))
eigAB= np.zeros(shape=(n_k,nspin,psiA.shape[2]+psiB.shape[2]))

#-----------------Construction of the coefficient matrix------------------------------
#Write the part of fragment A to the eigenvector matrix
for nk in range(n_k):
    for ns in range(nspin):
        for nst in range(psiA.shape[2]):
            for nb in range(psiA.shape[3]):
                psi1 = psiA[nk,ns,nst,nb]
                CA[nst,nb]=psi1
                CAB[nk,ns,nst,nb]=psi1#*(0.13)
                CCAB[nk,ns,nb,nst]=psi1
                occAB[nk,ns,nst]=occA[nk,ns,nst]
                eigAB[nk,ns,nst]=eigenvaluesA[nk,ns,nst]

#Write the part of fragment A to the eigenvector matrix                
for nk in range(n_k):
    for ns in range(nspin):
        for nst in range(psiB.shape[2]):
            for nb in range(psiB.shape[3]):
                psi1 = psiB[nk,ns,nst,nb]
                CB[nst,nb]=psi1
                CAB[nk,ns,nst+psiA.shape[2],nb]=psi1
                CCAB[nk,ns,nb+psiA.shape[3],nst+psiA.shape[2]]=psi1
                occAB[nk,ns,nst+occA.shape[2]]=occB[nk,ns,nst]
                eigAB[nk,ns,nst+psiA.shape[2]]=eigenvaluesB[nk,ns,nst]
                

CC_aims = np.zeros(shape=(n_k,nspin,psiA.shape[3]+psiB.shape[3],psiA.shape[2]+psiB.shape[2]),dtype='complex128')
for nk in range(n_k):
    for ns in range(nspin):       
            cc_aims = np.transpose(CCAB[nk,ns,:,:])
            CC_aims[nk,ns,:,:] = cc_aims
#-----------------End of the Construction of the coefficient matrix------------------
                
N_k_control=n_k
reproduce = 0
path = '/home/christian/vsc/energy_decomposition/restart_routines/test_systems/H2_minimal/spin_collinear/AB_restart/'
write_restart_files_from_input(path,N_k_control,eigAB,CC_aims,occAB,orb_posAB,kpoint_weights,reproduce)              
    
#--------------------------------------------------------------------------------------------------------

        