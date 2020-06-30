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
import scipy.linalg as lalg
from write_restart import *
import os
#import gc

debug = False

if debug==False:
    #PP = '/home/christian/Dokumente/Home_office/energy_decomposition/Quinacridone/shift_008/basis/multipole/' 
    PP= '/home/christian/Dokumente/Home_office/COFs/COF-1/energy_decomposition/shift_35/multipole/'
    sp = False
    N_k_control=48
    reproduce = 0
    atoms = read(PP+'AB/geometry.in')
    cell = atoms.cell
    Ha_to_eV = 1/27.2114
    
    filename = PP+'AB/output.aims'
    
    fermi_level, kpoint_weights = aims_read_fermi_and_kpoints(filename, cell)
    
    nkpts = len(kpoint_weights)
    
    eigenvaluesAB, psiAB, occ_AB, orb_posAB = aims_read_eigenvalues_and_coefficients(fermi_level, PP+'AB', spin=sp, debug=False)
    
    params=np.zeros(shape=(N_k_control,5,psiAB.shape[1]),dtype='int32')
    params[:,0,0]=psiAB.shape[0]
    params[:,1,0]=psiAB.shape[3]
    params[:,2,0]=psiAB.shape[2]
    params[:,3,0]=psiAB.shape[1]
    params[:,4,0]=psiAB.shape[0]
    
    meta_AB=psiAB.shape
    del psiAB
    #gc.collect()
    
    eigenvaluesA, psiA, occA, orb_posA = aims_read_eigenvalues_and_coefficients(fermi_level, PP+'fragA', spin=sp, debug=False)
    eigenvaluesB, psiB, occB, orb_posB = aims_read_eigenvalues_and_coefficients(fermi_level, PP+'fragB', spin=sp, debug=False)
    
    del orb_posA
    del orb_posB
    del orb_posAB
    
    #gc.collect()
    #
    #
    #REPRODUCE_RESTART_FILES(PP+'AB_test/',psiAB,eigenvaluesAB*Ha_to_eV,occ_AB,N_k_control,params,reproduce=1)
    
    
    HAB, SAB = aims_read_HS(PP+'AB',spin=sp)
    HA, SA = aims_read_HS(PP+'fragA',spin=sp)
    HB, SB = aims_read_HS(PP+'fragB',spin=sp)
    
    del HAB
    del HA
    del HB
    del SA
    del SB
    #gc.collect()
    
    nspin = meta_AB[1] #number-1
    n_k = nkpts   #number-1
    
    #Building the coefficient matrices of the fragments CA, CB and the combined system CAB
    CA = np.zeros(shape=(psiA.shape[2],psiA.shape[3]),dtype='complex128')
    CB = np.zeros(shape=(psiB.shape[2],psiB.shape[3]),dtype='complex128')
    
    CAB = np.zeros(shape=(n_k,nspin,psiA.shape[2]+psiB.shape[2],psiA.shape[3]+psiB.shape[3]),dtype='complex128')
    CCAB = np.zeros(shape=(n_k,nspin,meta_AB[2],meta_AB[3]),dtype='complex128')
    occAB= np.zeros(shape=(n_k,nspin,psiA.shape[2]+psiB.shape[2]))
    eigAB= np.zeros(shape=(n_k,nspin,psiA.shape[2]+psiB.shape[2]))
elif debug==True:
    print("!!! Debugging mode has been enabled !!!")
    print("Test system will be set up!")
    n_k=1
    nspin=1
    psiA=np.array([[[[1.,2.],[3.,4.]]]])
    occA=np.array([[[2.,2.]]])
    eigenvaluesA=np.array([[[-11.,5.]]])
    psiB=np.array([[[[5.,6.],[7.,8.]]]])
    occB=np.array([[[2.,0.]]])
    eigenvaluesB=np.array([[[-11.,5.]]])
    SAB=np.array([[[1.,0.,0.2,0.45],[0.,1.,0.45,0.35],[0.2,0.45,1.,0.],[0.45,0.35,0.,1.]]])
    
    CA = np.zeros(shape=(psiA.shape[2],psiA.shape[3]),dtype='complex128')
    CB = np.zeros(shape=(psiB.shape[2],psiB.shape[3]),dtype='complex128')
    
    CAB = np.zeros(shape=(n_k,nspin,psiA.shape[2]+psiB.shape[2],psiA.shape[3]+psiB.shape[3]),dtype='complex128')
    CCAB = np.zeros(shape=(n_k,nspin,psiA.shape[3]+psiB.shape[3],psiA.shape[2]+psiB.shape[2]),dtype='complex128')
    occAB= np.zeros(shape=(n_k,nspin,psiA.shape[2]+psiB.shape[2]))
    eigAB= np.zeros(shape=(n_k,nspin,psiA.shape[2]+psiB.shape[2]))
    psiAB=eigAB
    occ_AB=occAB


#-----------------Construction of the coefficient matrix-----------------------
#           ( C_occ_A       0     )
#    CAB =  ( C_virt_A      0     )
#           (   0        C_occ_B  )
#           (   0        C_virt_B )
#    occAB = [occ_A, virt_A, occ_B, virt_B]

#Write the part of fragment A to the eigenvector matrix
for nk in range(n_k):
    for ns in range(nspin):
        for nst in range(psiA.shape[2]):
            for nb in range(psiA.shape[3]):
                psi1 = psiA[nk,ns,nst,nb]
                CA[nst,nb]=psi1
                CAB[nk,ns,nst,nb]=psi1#*(0.13)
                #CCAB[nk,ns,nb,nst]=psiAB[nk,ns,nst,nb]
                occAB[nk,ns,nst]=occA[nk,ns,nst]
                eigAB[nk,ns,nst]=eigenvaluesA[nk,ns,nst]

#Write the part of fragment B to the eigenvector matrix                
for nk in range(n_k):
    for ns in range(nspin):
        for nst in range(psiB.shape[2]):
            for nb in range(psiB.shape[3]):
                psi1 = psiB[nk,ns,nst,nb]
                CB[nst,nb]=psi1
                CAB[nk,ns,nst+psiA.shape[2],nb+psiA.shape[3]]=psi1
                #CCAB[nk,ns,nb+psiA.shape[3],nst+psiA.shape[2]]=psiAB[nk,ns,nst,nb]
                occAB[nk,ns,nst+psiA.shape[2]]=occB[nk,ns,nst]
                eigAB[nk,ns,nst+psiA.shape[2]]=eigenvaluesB[nk,ns,nst]
##Manipulation to test how switching the states affects the restart functionality
# for nk in range(n_k):
#     for ns in range(nspin):
#         for nst in range(meta_AB[2]):
#             for nb in range(meta_AB[3]):
#                 CCAB[nk,ns,nst,nb] = psiAB[nk,ns,nst,nb]
#for nk in range(n_k):
#    for ns in range(nspin):
#        eigenvaluesAB[nk,ns,0:9] = np.flip(eigenvaluesAB[nk,ns,0:9],0) 
#        occ_AB[nk,ns,:] = np.flip(occ_AB[nk,ns,:],0) 
#        eigenvaluesAB[nk,ns,0]=eigenvaluesAB[nk,ns,0]*(-1.)
#        occ_AB[nk,ns,0] = 0.
#        for nb in range(psiAB.shape[3]):  
#            CCAB[nk,ns,0:9,nb] = np.flip(CCAB[nk,ns,0:9,nb],0)
           
#---------------Grouping occupied and virtual orbitals of fragments A, B-------
#              ( C_occ_A       0     )
#              (   0        C_occ_B  )
#    CAB_ov =  ( C_virt_A      0     )
#              (   0        C_virt_B )

#n_occ_A=occA.shape[2]
if nspin>1:
    ###### UNDER CONSTRUCTION########
    # So far only for symmetrical spin distribution meaning n_up = n_down !!! #
    n_up_A=np.count_nonzero(occA[0][0][:])
    n_up_B=np.count_nonzero(occB[0][0][:])
    n_up_total = n_up_A + n_up_B
    n_virt_up = occAB.shape[2] - np.count_nonzero(occAB[0][0][:])
    
    n_down_A=np.count_nonzero(occA[0][1][:])
    n_down_B=np.count_nonzero(occB[0][1][:])
    n_down_total = n_down_A + n_down_B
    n_virt_down = occAB.shape[2] - np.count_nonzero(occAB[0][1][:])

    occ_AB_FO = 0*occ_AB
    CAB_o_up = np.zeros(shape=(n_k,1,n_up_total,psiA.shape[3]+psiB.shape[3]),dtype='complex128')
    CAB_o_down = np.zeros(shape=(n_k,1,n_down_total,psiA.shape[3]+psiB.shape[3]),dtype='complex128')
    CAB_v_up = np.zeros(shape=(n_k,1,n_virt_up,psiA.shape[3]+psiB.shape[3]),dtype='complex128')
    CAB_v_down = np.zeros(shape=(n_k,1,n_virt_down,psiA.shape[3]+psiB.shape[3]),dtype='complex128')
    CAB_o = np.zeros(shape=(n_k,nspin,n_up_total,psiA.shape[3]+psiB.shape[3]),dtype='complex128')
    CAB_v_ll = np.zeros(shape=(n_k,nspin,n_virt_up,psiA.shape[3]+psiB.shape[3]),dtype='complex128')
    for nk in range(n_k):
        for ns in range(nspin):
            n_o_up=0
            n_o_down=0
            n_v_up=0
            n_v_down=0
            n_occ = 0
            n_v = 0
            n_vv=[]
            for nst in range(CAB.shape[2]):
                if occAB[nk,ns,nst]>1e-2:
                    if ns == 0:
                        CAB_o_up[nk,0,n_o_up,:]=CAB[nk,ns,nst,:]
                        CAB_o[nk,0,n_o_up,:]=CAB[nk,ns,nst,:]
                        occ_AB_FO[nk,ns,n_occ] = occAB[nk,ns,nst]
                        n_o_up+=1
                        n_occ+=1
                    else:
                        CAB_o_down[nk,0,n_o_down,:]=CAB[nk,ns,nst,:]
                        CAB_o[nk,1,n_o_down,:]=CAB[nk,ns,nst,:]
                        occ_AB_FO[nk,ns,n_occ] = occAB[nk,ns,nst]
                        n_o_down+=1
                        n_occ+=1
                elif nst<meta_AB[2]:
                    if ns == 0:
                        CAB_v_up[nk,0,n_v_up,:]=CAB[nk,ns,nst,:]
                        CAB_v_ll[nk,0,n_v_up,:]=CAB[nk,ns,nst,:]
                        n_v_up+=1
                        n_v+=1
                    else:
                        CAB_v_down[nk,0,n_v_down,:]=CAB[nk,ns,nst,:]
                        CAB_v_ll[nk,1,n_v_down,:]=CAB[nk,ns,nst,:]
                        n_v_down+=1
                        n_v+=1
            n_vv.append(n_v)
    if all(x == n_vv[0] for x in n_vv):
        CAB_v = CAB_v_ll[:,:,0:n_v,:]
    else:
        print("Not all k-points and spins have the same number of virtual states!")
    
    n_occ_A = CAB_o.shape[2]
    n_occ_B = 0
    
else:
    n_occ_A=np.count_nonzero(occA[0][0][:])
    n_occ_B=np.count_nonzero(occB[0][0][:])   
    n_virt_A=occA.shape[2]-n_occ_A#occA.shape[2]-min(np.count_nonzero(occA[0][0][:]),np.count_nonzero(occA[0][1][:]))#occA.shape[2]-n_occ_A
    n_virt_B=occB.shape[2]-n_occ_B#occB.shape[2]-min(np.count_nonzero(occB[0][0][:]),np.count_nonzero(occB[0][1][:]))#occB.shape[2]-n_occ_B

    occ_AB_FO = 0*occ_AB
    CAB_o = np.zeros(shape=(n_k,nspin,n_occ_A+n_occ_B,psiA.shape[3]+psiB.shape[3]),dtype='complex128')
    #CAB_v = np.zeros(shape=(n_k,nspin,n_virt_A+n_virt_B,psiA.shape[3]+psiB.shape[3]),dtype='complex128')
    CAB_v_ll = np.zeros(shape=(n_k,nspin,meta_AB[2]-n_occ_A-n_occ_B,psiA.shape[3]+psiB.shape[3]),dtype='complex128')
    
    n_vv=[]
    for nk in range(n_k):
#        if nk in range(nkpts+1)[1:]:
#            k_valid = 1
#        else:
#            k_valid = 0
#            
#        if k_valid:
        for ns in range(nspin):
            n_o=0
            n_v=0
            for nst in range(CAB.shape[2]):
                if occAB[nk,ns,nst]>1e-8:
                    CAB_o[nk,ns,n_o,:]=CAB[nk,ns,nst,:]
                    #CAB_ov[nk,ns,n_o,:]=CAB[nk,ns,nst,:]
                    occ_AB_FO[nk,ns,n_o] = occAB[nk,ns,nst]
                    n_o+=1
                elif nst<meta_AB[2]:
                    CAB_v_ll[nk,ns,n_v,:]=CAB[nk,ns,nst,:]
                    #eigenvaluesAB[nk,ns,n_occ_A+n_occ_B+n_v] = np.abs(eigenvaluesAB[nk,ns,n_occ_A+n_occ_B+n_v])
                    #CAB_ov[nk,ns,n_occ_A+n_occ_B+n_v,:]=CAB[nk,ns,nst,:]
                    n_v+=1
            n_vv.append(n_v)
CAB_v = CAB_v_ll            
#    if all(x == n_vv[0] for x in n_vv):
#        CAB_v = CAB_v_ll[:,:,0:n_v,:]
#    else:
#        print("Not all k-points and spins have the same number of virtual states!")

CAB_ov = np.zeros(shape=(n_k,nspin,CAB_o.shape[2]+CAB_v.shape[2],psiA.shape[3]+psiB.shape[3]),dtype='complex128')
CAB_ov[:,:,0:CAB_o.shape[2],:] = CAB_o
CAB_ov[:,:,CAB_o.shape[2]:,:] = CAB_v
CAB_wo_transform = CAB_ov

path = PP+'AB_wo_transform/'
if not os.path.exists(path):
    os.makedirs(path)
REPRODUCE_RESTART_FILES(path,CAB_wo_transform,eigenvaluesAB*Ha_to_eV,occ_AB,N_k_control,params,reproduce=1)


        