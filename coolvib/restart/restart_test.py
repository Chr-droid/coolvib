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

debug = False

if debug==False:
    PP = '/home/christian/vsc/energy_decomposition/restart_routines/test_systems/Quinacridone/no_shift/more_k/'
    sp = False
    atoms = read(PP+'AB/geometry.in')
    cell = atoms.cell
    
    filename = PP+'AB/output.aimsrestart'
    
    fermi_level, kpoint_weights = aims_read_fermi_and_kpoints(filename, cell)
    
    nkpts = len(kpoint_weights)
    
    eigenvaluesAB, psiAB, occ_AB, orb_posAB = aims_read_eigenvalues_and_coefficients(fermi_level, PP+'AB', spin=sp, debug=False)
    eigenvaluesA, psiA, occA, orb_posA = aims_read_eigenvalues_and_coefficients(fermi_level, PP+'fragA', spin=sp, debug=False)
    eigenvaluesB, psiB, occB, orb_posB = aims_read_eigenvalues_and_coefficients(fermi_level, PP+'fragB', spin=sp, debug=False)
    #----------- eigenvalues = np.zeros([n_kpts, n_spin, n_states])
    #----------- occ = np.zeros([n_kpts, n_spin, n_states])
    #----------- psi = np.zeros([n_kpts, n_spin, n_states, n_basis],dtype=complex)
    #----------- orbital_pos = np.zeros(n_basis,dtype=np.int)
    
    HAB, SAB = aims_read_HS(PP+'AB',spin=sp)
    HA, SA = aims_read_HS(PP+'fragA',spin=sp)
    HB, SB = aims_read_HS(PP+'fragB',spin=sp)
    
    nspin = 1 #number-1
    n_k = nkpts    #number-1
    
    #Building the coefficient matrices of the fragments CA, CB and the combined system CAB
    CA = np.zeros(shape=(psiA.shape[2],psiA.shape[3]),dtype='complex128')
    CB = np.zeros(shape=(psiB.shape[2],psiB.shape[3]),dtype='complex128')
    
    CAB = np.zeros(shape=(n_k,nspin,psiA.shape[2]+psiB.shape[2],psiA.shape[3]+psiB.shape[3]),dtype='complex128')
    CCAB = np.zeros(shape=(n_k,nspin,psiAB.shape[2],psiAB.shape[3]),dtype='complex128')
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
#for nk in range(n_k):
#    for ns in range(nspin):
#        for nst in range(psiAB.shape[2]):
#            for nb in range(psiAB.shape[3]):
#                CCAB[nk,ns,nst,nb] = psiAB[nk,ns,nst,nb]
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
    for nk in range(n_k):
        for ns in range(nspin):
            n_o_up=0
            n_o_down=0
            n_v_up=0
            n_v_down=0
            n_occ = 0
            for nst in range(CAB.shape[2]):
                if occAB[nk,ns,nst]>1e-2:
                    if ns == 0:
                        CAB_o_up[nk,0,n_o_up,:]=CAB[nk,ns,nst,:]
                        occ_AB_FO[nk,ns,n_occ] = occAB[nk,ns,nst]
                        n_o_up+=1
                        n_occ+=1
                    else:
                        CAB_o_down[nk,0,n_o_down,:]=CAB[nk,ns,nst,:]
                        occ_AB_FO[nk,ns,n_occ] = occAB[nk,ns,nst]
                        n_o_down+=1
                        n_occ+=1
                elif nst<psiAB.shape[2]:
                    if ns == 0:
                        CAB_v_up[nk,0,n_v_up,:]=CAB[nk,ns,nst,:]
                        n_v_up+=1
                    else:
                        CAB_v_down[nk,0,n_v_down,:]=CAB[nk,ns,nst,:]
                        n_v_down+=1
else:
    n_occ_A=np.count_nonzero(occA[0][0][:])
    n_occ_B=np.count_nonzero(occB[0][0][:])   
n_virt_A=occA.shape[2]-n_occ_A#occA.shape[2]-min(np.count_nonzero(occA[0][0][:]),np.count_nonzero(occA[0][1][:]))#occA.shape[2]-n_occ_A
n_virt_B=occB.shape[2]-n_occ_B#occB.shape[2]-min(np.count_nonzero(occB[0][0][:]),np.count_nonzero(occB[0][1][:]))#occB.shape[2]-n_occ_B

occ_AB_FO = 0*occ_AB
CAB_o = np.zeros(shape=(n_k,nspin,n_occ_A+n_occ_B,psiA.shape[3]+psiB.shape[3]),dtype='complex128')
#CAB_v = np.zeros(shape=(n_k,nspin,n_virt_A+n_virt_B,psiA.shape[3]+psiB.shape[3]),dtype='complex128')
CAB_v_ll = np.zeros(shape=(n_k,nspin,n_virt_A+n_virt_B,psiA.shape[3]+psiB.shape[3]),dtype='complex128')
n_vv=[]
for nk in range(n_k):
    for ns in range(nspin):
        n_o=0
        n_v=0
        for nst in range(CAB.shape[2]):
            if occAB[nk,ns,nst]>1e-8:
                CAB_o[nk,ns,n_o,:]=CAB[nk,ns,nst,:]
                #CAB_ov[nk,ns,n_o,:]=CAB[nk,ns,nst,:]
                occ_AB_FO[nk,ns,n_o] = occAB[nk,ns,nst]
                n_o+=1
            elif nst<psiAB.shape[2]:
                CAB_v_ll[nk,ns,n_v,:]=CAB[nk,ns,nst,:]
                #CAB_ov[nk,ns,n_occ_A+n_occ_B+n_v,:]=CAB[nk,ns,nst,:]
                n_v+=1
        n_vv.append(n_v)
        
if all(x == n_vv[0] for x in n_vv):
    CAB_v = CAB_v_ll[:,:,0:n_v,:]
else:
    print("Not all k-points and spins have the same number of virtual states!")

CAB_ov = np.zeros(shape=(n_k,nspin,n_occ_A+n_occ_B+CAB_v.shape[2],psiA.shape[3]+psiB.shape[3]),dtype='complex128')
CAB_ov[:,:,0:CAB_o.shape[2],:] = CAB_o
CAB_ov[:,:,CAB_o.shape[2]:,:] = CAB_v
CAB_wo_transform = CAB_ov

##Normalize occupied states################# EXPERIMENTAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!
#n_v=0
#for nk in range(n_k):
#    for ns in range(nspin):
#        for nst in range(CAB_ov.shape[2]):
#            normal_occ = np.sum(CAB_ov[nk,ns,nst,:]**2)
#            CAB_ov[nk,ns,nst,:] = CAB_ov[nk,ns,nst,:]*(1/normal_occ)
#            if nst< n_o:
#                CAB_o[nk,ns,nst,:] = CAB_o[nk,ns,nst,:]*(1/normal_occ)
#            elif n_v<CAB_v.shape[2]:
#                CAB_v[nk,ns,n_v,:] = CAB_v[nk,ns,n_v,:]*(1/normal_occ) 
#                n_v+=1

#---------Transform the overlap matrix SAB into the fragment orbital basis-----
#       SAB_o/o_FO = CAB_o(k,spin) * SAB(k,spin) * (CAB_o(k,spin))^T
#       SAB_v/v_FO = CAB_v(k,spin) * SAB(k,spin) * (CAB_v(k,spin))^T
#       SAB_o/v_FO = CAB_o(k,spin) * SAB(k,spin) * (CAB_v(k,spin))^T
#       SAB_v/o_FO = CAB_v(k,spin) * SAB(k,spin) * (CAB_o(k,spin))^T

#TRY --- Also build the overlap matrix consisting of the overlap matrices of the fragments SAB = [[SAB_A,0],[0,SAB_B]]
#S_AB=0*SAB
#S_AB[:,0:SA.shape[1],0:SA.shape[2]]=SA
#S_AB[:,SA.shape[1]:,SA.shape[2]:]=SB
#SAB=S_AB
SAB_oo = np.zeros(shape=(n_k,nspin,n_occ_A+n_occ_B,n_occ_A+n_occ_B),dtype='complex128')
SAB_vv = np.zeros(shape=(n_k,nspin,CAB_v.shape[2],CAB_v.shape[2]),dtype='complex128')
SAB_ov = np.zeros(shape=(n_k,nspin,n_occ_A+n_occ_B,CAB_v.shape[2]),dtype='complex128')
SAB_vo = np.zeros(shape=(n_k,nspin,CAB_v.shape[2],n_occ_A+n_occ_B),dtype='complex128')
SAB_all = np.zeros(shape=(n_k,nspin,n_occ_A+n_occ_B+CAB_v.shape[2],n_occ_A+n_occ_B+CAB_v.shape[2]),dtype='complex128')
for nk in range(n_k):
    for ns in range(nspin):
        ######### SAB ########
        CAB_T = np.matrix.conjugate(np.transpose(CAB_ov[nk,ns,:,:]))
        P = np.matmul(CAB_ov[nk,ns,:,:],SAB[nk,:,:])
        SAB_FO = np.matmul(P,CAB_T)
        SAB_all[nk,ns,:,:] = SAB_FO
        ####### SAB_oo #######
        CAB_T = np.matrix.conjugate(np.transpose(CAB_o[nk,ns,:,:]))
        P = np.matmul(CAB_o[nk,ns,:,:],SAB[nk,:,:])
        SAB_FO = np.matmul(P,CAB_T)
        SAB_oo[nk,ns,:,:] = SAB_FO
        ####### SAB_vv #######
        CAB_T = np.matrix.conjugate(np.transpose(CAB_v[nk,ns,:,:]))
        P = np.matmul(CAB_v[nk,ns,:,:],SAB[nk,:,:])
        SAB_FO = np.matmul(P,CAB_T)
        SAB_vv[nk,ns,:,:] = SAB_FO
        ####### SAB_ov #######
        CAB_T = np.matrix.conjugate(np.transpose(CAB_v[nk,ns,:,:]))
        P = np.matmul(CAB_o[nk,ns,:,:],SAB[nk,:,:])
        SAB_FO = np.matmul(P,CAB_T)
        SAB_ov[nk,ns,:,:] = SAB_FO
        ####### SAB_vo #######
        CAB_T = np.matrix.conjugate(np.transpose(CAB_o[nk,ns,:,:]))
        P = np.matmul(CAB_v[nk,ns,:,:],SAB[nk,:,:])
        SAB_FO = np.matmul(P,CAB_T)
        SAB_vo[nk,ns,:,:] = SAB_FO
#-------End of the construction of the overlap matrices in FO basis------------

#-------Orthogonalize occ fragment orbitals relative to each other-------------
#       CAB_oo_FO1 = (SAB_oo)^(-1/2) * CAB_o
CAB_oo_FO1 = np.zeros(shape=(n_k,nspin,n_occ_A+n_occ_B,psiA.shape[3]+psiB.shape[3]),dtype='complex128')
SAB_oo_ii_vec = np.zeros(shape=(n_k,nspin,n_occ_A+n_occ_B,n_occ_A+n_occ_B),dtype='complex128')

for nk in range(n_k):
    for ns in range(nspin):
        ## Getting the ^(-1/2) of the overlap matrix
        lam_s, l_s = np.linalg.eigh(SAB_oo[nk,ns,:,:])
        lam_s = lam_s * np.eye(len(lam_s))
        lam_sqrt_inv = np.sqrt(np.linalg.inv(lam_s))
        symm_orthog = np.dot(l_s, np.dot(lam_sqrt_inv, l_s.T))
        SAB_oo_ii = symm_orthog # Compared the results to MATLAB - they agree!
        #SAB_oo_ii = np.linalg.inv(lalg.sqrtm(SAB_oo[nk,ns,:,:]))
        SAB_oo_ii_vec[nk,ns,:,:] = SAB_oo_ii
        ####################
        C_o_nn = np.matmul(SAB_oo_ii,CAB_o[nk,ns,:,:])#np.matmul(CAB_o[nk,ns,:,:],SAB_oo_ii)
        CAB_oo_FO1[nk,ns,:,:] = C_o_nn
print("STEP 1:   Loewdin Orthogonalization of occupied states")
#-------End of orthogonalization of occ fragment orbitals----------------------


#------- Construct the new overlap matrix SAB_oo_FO1 --------------------------
#       SAB_o/o_FO1 = CAB_o_FO1(k,spin) * SAB(k,spin) * (CAB_o_FO1(k,spin))^T
#       SAB_v/v_FO1 = CAB_v_FO1(k,spin) * SAB(k,spin) * (CAB_v_FO1(k,spin))^T
#       SAB_o/v_FO1 = CAB_o_FO1(k,spin) * SAB(k,spin) * (CAB_v_FO1(k,spin))^T
#       SAB_v/o_FO1 = CAB_v_FO1(k,spin) * SAB(k,spin) * (CAB_o_FO1(k,spin))^T            
SAB_oo_FO1 = np.zeros(shape=(n_k,nspin,n_occ_A+n_occ_B,n_occ_A+n_occ_B),dtype='complex128')
SAB_vv_FO1 = np.zeros(shape=(n_k,nspin,CAB_v.shape[2],CAB_v.shape[2]),dtype='complex128')
SAB_ov_FO1 = np.zeros(shape=(n_k,nspin,n_occ_A+n_occ_B,CAB_v.shape[2]),dtype='complex128')
SAB_vo_FO1 = np.zeros(shape=(n_k,nspin,CAB_v.shape[2],n_occ_A+n_occ_B),dtype='complex128')
SAB_all_FO1 = np.zeros(shape=(n_k,nspin,n_occ_A+n_occ_B+CAB_v.shape[2],n_occ_A+n_occ_B+CAB_v.shape[2]),dtype='complex128')
for nk in range(n_k):
    for ns in range(nspin):
        ######### SAB ########
#        CAB_T = np.matrix.conjugate(np.transpose(CAB_ov[nk,ns,:,:]))
#        P = np.matmul(CAB_ov[nk,ns,:,:],SAB[nk,:,:])
#        SAB_FO = np.matmul(P,CAB_T)
#        SAB_all_FO1[nk,ns,:,:] = SAB_FO
        ####### SAB_oo #######
        CAB_T = np.matrix.conjugate(np.transpose(CAB_oo_FO1[nk,ns,:,:]))
        P = np.matmul(CAB_oo_FO1[nk,ns,:,:],SAB[nk,:,:])
        SAB_FO = np.matmul(P,CAB_T)
        SAB_oo_FO1[nk,ns,:,:] = SAB_FO
        ####### SAB_vv #######
        CAB_T = np.matrix.conjugate(np.transpose(CAB_v[nk,ns,:,:]))
        P = np.matmul(CAB_v[nk,ns,:,:],SAB[nk,:,:])
        SAB_FO = np.matmul(P,CAB_T)
        SAB_vv_FO1[nk,ns,:,:] = SAB_FO
        ####### SAB_ov #######
        CAB_T = np.matrix.conjugate(np.transpose(CAB_v[nk,ns,:,:]))
        P = np.matmul(CAB_oo_FO1[nk,ns,:,:],SAB[nk,:,:])
        SAB_FO = np.matmul(P,CAB_T)
        SAB_ov_FO1[nk,ns,:,:] = SAB_FO
        ####### SAB_vo #######
        CAB_T = np.matrix.conjugate(np.transpose(CAB_oo_FO1[nk,ns,:,:]))
        P = np.matmul(CAB_v[nk,ns,:,:],SAB[nk,:,:])
        SAB_FO = np.matmul(P,CAB_T)
        SAB_vo_FO1[nk,ns,:,:] = SAB_FO
        
#-------Check if the orhtogonalization from above worked-----------------------
for nk in range(n_k):
    for ns in range(nspin):
        CAB_T = np.matrix.conjugate(np.transpose(SAB_oo_ii_vec[nk,ns,:,:]))
        CC = np.matmul(CAB_T,SAB_oo[nk,ns,:,:])
        MM = np.matmul(CC,SAB_oo_ii_vec[nk,ns,:,:])
        I=np.eye(MM.shape[0])
        if np.all(np.abs((np.abs(MM)-I))<1e-10) and np.all(np.abs(np.abs(SAB_oo_FO1[nk,ns,:,:])-I)<1e-8):
            print("STEP 1:   Loewdin Orthogonalization worked")
        else:
            print("STEP 1:   ERROR: Loewdin Orthogonalization did not work")

#-------Orthogonalize virt fragment orbitals with respect to occ ones----------
#       C_v_FO2 = CAB_v - C_corr
print("STEP 2:   Gram-Schmidt Orthogonalization of unoccupied states to occupied states")
C_GS = np.zeros(shape=(n_k,nspin,SAB_vo_FO1[nk,ns,:,:].shape[0],SAB_vo_FO1[nk,ns,:,:].shape[1]),dtype='complex128')
C_corr = np.zeros(shape=(n_k,nspin,SAB_vo_FO1[nk,ns,:,:].shape[0],CAB_oo_FO1[nk,ns,:,:].shape[1]),dtype='complex128')
C_v_FO2 = np.zeros(shape=(n_k,nspin,CAB_v[nk,ns,:,:].shape[0],CAB_v[nk,ns,:,:].shape[1]),dtype='complex128')

for nk in range(n_k):
    for ns in range(nspin):
        C_GS[nk,ns,:,:] = SAB_vo_FO1[nk,ns,:,:]#-1.*np.matmul(C_nn,SAB_ov_FO1[nk,ns,:,:])
        C_corr[nk,ns,:,:] = np.matmul(C_GS[nk,ns,:,:],CAB_oo_FO1[nk,ns,:,:])
        C_v_FO2[nk,ns,:,:] = CAB_v[nk,ns,:,:] - C_corr[nk,ns,:,:]
#-------Check if the orhtogonalization from above worked-----------------------
SAB_ov_FO2 = np.zeros(shape=(n_k,nspin,n_occ_A+n_occ_B,CAB_v.shape[2]),dtype='complex128')
SAB_vo_FO2 = np.zeros(shape=(n_k,nspin,CAB_v.shape[2],n_occ_A+n_occ_B),dtype='complex128')
SAB_vv_FO2 = np.zeros(shape=(n_k,nspin,CAB_v.shape[2],CAB_v.shape[2]),dtype='complex128')
for nk in range(n_k):
    for ns in range(nspin):
        ####### SAB_ov #######
        CAB_T = np.matrix.conjugate(np.transpose(C_v_FO2[nk,ns,:,:]))
        P = np.matmul(CAB_oo_FO1[nk,ns,:,:],SAB[nk,:,:])
        SAB_FO = np.matmul(P,CAB_T)
        SAB_ov_FO2[nk,ns,:,:] = SAB_FO
        ####### SAB_vo #######
        CAB_T = np.matrix.conjugate(np.transpose(CAB_oo_FO1[nk,ns,:,:]))
        P = np.matmul(C_v_FO2[nk,ns,:,:],SAB[nk,:,:])
        SAB_FO = np.matmul(P,CAB_T)
        SAB_vo_FO2[nk,ns,:,:] = SAB_FO
        ####### SAB_vv #######
        CAB_T = np.matrix.conjugate(np.transpose(C_v_FO2[nk,ns,:,:]))
        P = np.matmul(C_v_FO2[nk,ns,:,:],SAB[nk,:,:])
        SAB_FO = np.matmul(P,CAB_T)
        SAB_vv_FO2[nk,ns,:,:] = SAB_FO
        if np.all(np.abs(SAB_vo_FO2[nk,ns,:,:].real)<1e-10) and np.all(np.abs(SAB_ov_FO2[nk,ns,:,:].real)<1e-10):
            print("STEP 2:   Gram-schmidt Orthogonalization worked") 
        else:
            print("STEP 2:   ERROR: Gram-Schmidt Orthogonalization did not work")
#------- End of Orthogonalize virt fragment orbitals with respect to occ ones--
            
#-------Orthogonalize virt fragment orbitals relative to each other-------------
#       CAB_virt_FO1 = (SAB_vv)^(-1/2) * CAB_v
CAB_vv_FO3 = np.zeros(shape=(n_k,nspin,CAB_v.shape[2],psiA.shape[3]+psiB.shape[3]),dtype='complex128')
SAB_vv_FO3 = np.zeros(shape=(n_k,nspin,CAB_v.shape[2],CAB_v.shape[2]),dtype='complex128')
SAB_vv_FO3T = np.zeros(shape=(n_k,nspin,CAB_v.shape[2],CAB_v.shape[2]),dtype='complex128')

for nk in range(n_k):
    for ns in range(nspin):
#        SAB_oo_ii = lalg.sqrtm(SAB_oo[nk,ns,:,:])
        # Getting the ^(-1/2) of the overlap matrix
        lam_s, l_s = np.linalg.eigh(SAB_vv_FO2[nk,ns,:,:])
        lam_s = lam_s * np.eye(len(lam_s))
        lam_sqrt_inv = np.sqrt(np.linalg.inv(lam_s))
        symm_orthog = np.dot(l_s, np.dot(lam_sqrt_inv, l_s.T))
        SAB_oo_ii = symm_orthog
        SAB_vv_FO3T[nk,ns,:,:] = SAB_oo_ii
        ####################
        C_o_nn = np.matmul(SAB_oo_ii,C_v_FO2[nk,ns,:,:])
        CAB_vv_FO3[nk,ns,:,:] = C_o_nn
print("STEP 3:   Loewdin Orthogonalization of occupied states")
#-------End of orthogonalization of occ fragment orbitals----------------------

#-------Check if the orhtogonalization from above worked-----------------------
for nk in range(n_k):
    for ns in range(nspin):
        CAB_T = np.matrix.conjugate(np.transpose(SAB_vv_FO3T[nk,ns,:,:]))
        CC = np.matmul(CAB_T,SAB_vv_FO2[nk,ns,:,:])
        MM = np.matmul(CC,SAB_vv_FO3T[nk,ns,:,:])
        I=np.eye(MM.shape[0])
        ####### SAB_vv #######
        CAB_T = np.matrix.conjugate(np.transpose(CAB_vv_FO3[nk,ns,:,:]))
        P = np.matmul(CAB_vv_FO3[nk,ns,:,:],SAB[nk,:,:])
        SAB_FO = np.matmul(P,CAB_T)
        SAB_vv_FO3[nk,ns,:,:] = SAB_FO
        if np.all(np.abs(MM.real-I)<1e-10) and np.all(np.abs(SAB_FO.real-I)<1e-9):
            print("STEP 3:   Loewdin Orthogonalization worked nk="+str(nk))
        else:
            print("STEP 3:   ERROR: Loewdin Orthogonalization did not work nk="+str(nk))
#------- END Check if the orhtogonalization from above worked------------------
            
#--------- Build final overlap and coefficient matrix--------------------------
            
SAB_final = np.zeros(shape=(n_k,nspin,n_occ_A+n_occ_B+CAB_v.shape[2],n_occ_A+n_occ_B+CAB_v.shape[2]),dtype='complex128')
CAB_final = np.zeros(shape=(n_k,nspin,n_occ_A+n_occ_B+CAB_v.shape[2],psiA.shape[3]+psiB.shape[3]),dtype='complex128')
for nk in range(n_k):
    for ns in range(nspin):
        SAB_final[nk,ns,0:n_occ_A+n_occ_B,0:n_occ_A+n_occ_B] = SAB_oo_FO1[nk,ns,:,:]
        SAB_final[nk,ns,0:n_occ_A+n_occ_B,n_occ_A+n_occ_B:] = SAB_ov_FO2[nk,ns,:,:]
        SAB_final[nk,ns,n_occ_A+n_occ_B:,0:n_occ_A+n_occ_B] = SAB_vo_FO2[nk,ns,:,:]
        SAB_final[nk,ns,n_occ_A+n_occ_B:,n_occ_A+n_occ_B:] = SAB_vv_FO3[nk,ns,:,:]
        CAB_final[nk,ns,0:n_occ_A+n_occ_B,0:] = CAB_oo_FO1[nk,ns,:,:]#*0
        CAB_final[nk,ns,n_occ_A+n_occ_B:,0:] = CAB_vv_FO3[nk,ns,:,:]#*0.001
        
##Normalize occupied and virtual states################# EXPERIMENTAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!
#for nk in range(n_k):
#    for ns in range(nspin):
#        for nst in range(CAB_ov.shape[2]):
#            normal_occ = np.sum(CAB_final[nk,ns,nst,:]**2)
#            CAB_final[nk,ns,nst,:] = CAB_final[nk,ns,nst,:]*(1/normal_occ)
            
#-----------------End of the Construction of the coefficient matrix------------------
                
N_k_control=75
reproduce = 0
path = PP+'AB_restart/'
write_restart_files_from_input(path,N_k_control,eigenvaluesAB,CAB_final,occ_AB_FO,orb_posAB,kpoint_weights,reproduce)

path = PP+'AB_wo_transform/'
write_restart_files_from_input(path,N_k_control,eigenvaluesAB,CAB_wo_transform,occ_AB_FO,orb_posAB,kpoint_weights,reproduce)

path = PP+'AB_test/'
#write_restart_files_from_input(path,N_k_control,eigenvaluesAB,psiAB,occ_AB,orb_posAB,kpoint_weights,reproduce)              
    
#--------------------------------------------------------------------------------------------------------

        