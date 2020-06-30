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
write_restart.py

This function can read and write restart files from the FHI-AIMS output.
"""

import numpy as np
from aims import *
from ase.io import *
from scipy.io import FortranFile
import restartutils
#from aimsutils.parser import parse_aimsout
import os
    
#--------------------------------------------------------------------------------------------------------
    


#N_k_control=16
reproduce = 0
path = '/home/christian/vsc/energy_decomposition/restart_routines/test_systems/H2/blank_test/'
def write_restart_files(path,N_k_control,reproduce=0):
    atoms = read(path+'geometry.in')
    cell = atoms.cell
    filename = path+'output.aimsrestart'
    fermi_level, kpoint_weights = aims_read_fermi_and_kpoints(filename, cell)
    nkpts = len(kpoint_weights)
    eigenvalues, psi, occ, orb_pos = aims_read_eigenvalues_and_coefficients(fermi_level, path[:-1], spin=False, debug=False)
    H, S = aims_read_HS(path[:-1],spin=False)
    KS_out = np.zeros(shape=(12,1*5*10),dtype='complex128') # hardcoded only for test cases!!!
    KS_out_grp = np.zeros(shape=(12,1,5,10),dtype='complex128')
    eig_grp = np.zeros(shape=(1,5,10),dtype='complex128')
    occ_grp = np.zeros(shape=(1,5,10),dtype='complex128')
    for ii in range(N_k_control):
        Ha_to_eV = 1./27.2114
        if reproduce:
            if ii<10:
                num = '00'+str(ii)
            else:
                num = '0'+str(ii)
            n1=path+'test'+num
            n2=path+'restart/'+'TTest'+num
            
            f = FortranFile(n2, 'w')
            f.close()
            
            if ii in range(nkpts+1)[1:]:
                k_valid = 1
            else:
                k_valid = 0
                
            with FortranFile(n1) as f:
                n_k_points = np.shape(kpoint_weights)[0]
                n_basis = f.read_record('i4')
                n_states_to_save = f.read_record('i4')
                n_spin = f.read_record('i4')
                n_k_points_task = f.read_record('i4')
                #print n_k_points_task
                KS_vec=[]
                occ_numbers = []
                KS_eigen = []
                KS_eigen_occ = []
                n_task_list = []
                
                if k_valid:
                    print(n_spin[0]*n_states_to_save[0]*n_basis[0])
                    print(n_spin[0],n_states_to_save[0],n_basis[0])
                    for i in range(n_spin[0]*n_states_to_save[0]*n_basis[0]):
                        e1 = f.read_record('f8')
                        #print(e1)
                        e2 = e1[0]+1j*e1[1]
                        if np.abs(e1[0])<1e-10:
                            KS_vec.append(e2)#*0.0) one could introduce a threshold
                        else:
                            KS_vec.append(e2)
                    #print(KS_vec)
                    KS_out[ii-1] = np.array(KS_vec) #np.array(KS_vec).reshape(n_basis[0],n_states_to_save[0], n_spin[0])
                    KS_out_grp[ii-1] = np.array(KS_vec).reshape(n_spin[0],n_states_to_save[0],n_basis[0])
                    for j in range(n_spin[0]*n_states_to_save[0]*n_k_points):
                        e3 = f.read_record('f8')
                        occ_numbers.append(np.array(e3[1]))
                        KS_eigen.append(np.array(e3[0]))
                        KS_eigen_occ.append(e3)
                else:
                    print(n_spin[0]*n_states_to_save[0]*n_basis[0])
                    print(n_spin[0],n_states_to_save[0],n_basis[0])
                    for j in range(n_spin[0]*n_states_to_save[0]*n_k_points):
                        #print(n_spin,n_states_to_save,n_basis,n_k_points)
                        e3 = f.read_record('f8')
                        occ_numbers.append(np.array(e3[1]))
                        KS_eigen.append(np.array(e3[0]))
                        KS_eigen_occ.append(e3)
                    print("Reading header files")
                
                eig_grp = np.array(KS_eigen).reshape(n_k_points,n_spin[0],n_states_to_save[0])
                occ_grp = np.array(KS_eigen).reshape(n_k_points,n_spin[0],n_states_to_save[0])
            
                
                    
            with FortranFile(n2,mode='w') as f:
                f.write_record(n_basis)
                f.write_record(n_states_to_save)
                f.write_record(n_spin)
                f.write_record(n_k_points_task)
                if k_valid:
                    for i in range(n_spin[0]*n_states_to_save[0]*n_basis[0]):
                        #print(KS_vec[i])
                        f.write_record(KS_vec[i])
                    for j in range(n_spin[0]*n_states_to_save[0]*n_k_points):
                        f.write_record(KS_eigen_occ[j])
                        #print(KS_eigen_occ[j])
                        #f.write_record(occ_numbers[j])
                else:
                    for j in range(n_spin[0]*n_states_to_save[0]*n_k_points):
                        f.write_record(KS_eigen_occ[j])
                    print("Writing header files")
        else:
            if ii<10:
                num = '00'+str(ii)
            else:
                num = '0'+str(ii)
            n1='Ttest'+num
            n2='Ttest'+num
            
            f = FortranFile(path+n2, 'w')
            f.close()
            
            if ii in range(nkpts+1)[1:]:
                k_valid = 1
            else:
                k_valid = 0
            
            with FortranFile(path+n2,mode='w') as f:
                n_k_points = np.shape(kpoint_weights)[0]
                n_basis = np.array([orb_pos.size],dtype='int32')
                n_states_to_save = np.array([occ.shape[2]],dtype='int32')
                n_spin = np.array([occ.shape[1]],dtype='int32')
                #n_k_points_task = np.array([1],dtype='int32') #hardcoded but could be read in from the output
                f.write_record(n_basis)
                f.write_record(n_states_to_save)
                f.write_record(n_spin)
                ll=0
                if k_valid:
                    n_k_points_task = np.array([1],dtype='int32')
                    f.write_record(n_k_points_task)#######
                    for i_k in range(n_k_points_task):
                        for i_spin in range(n_spin):
                            for i_states in range(n_states_to_save):
                                for i_basis in range(n_basis):
                                    ee1 = psi[i_k,i_spin,i_states,i_basis]
                                    ee = np.array([ee1.real,ee1.imag],dtype='float64')
                                    ll+=1
                                    f.write_record(ee)
                                    
                    for i_k in range(n_k_points):
                        for i_spin in range(n_spin):
                            for i_states in range(n_states_to_save):
                                ee2 = eigenvalues[i_k,i_spin,i_states]*Ha_to_eV
                                ee3 = occ[i_k,i_spin,i_states]
                                ee = np.array([ee2,ee3],dtype='float64')
                                f.write_record(ee)
                else:
                    n_k_points_task = np.array([0],dtype='int32')
                    f.write_record(n_k_points_task)#######
                    for i_k in range(n_k_points):
                        for i_spin in range(n_spin):
                            for i_states in range(n_states_to_save):
                                ee2 = eigenvalues[i_k,i_spin,i_states]*Ha_to_eV
                                ee3 = occ[i_k,i_spin,i_states]
                                ee = np.array([ee2,ee3],dtype='float64')
                                f.write_record(ee)
                    print("Printing header files")
    
#    return KS_out, KS_eigen, occ_numbers, KS_out_grp, eig_grp, occ_grp

def READ_RESTART_FILES(path,N_k_control,name='test',reproduce=1):
    atoms = read(path+'geometry.in')
    cell = atoms.cell
    filename = path+'output.aimsrestart'
    fermi_level, kpoint_weights = aims_read_fermi_and_kpoints(filename, cell)
    nkpts = len(kpoint_weights)
#    eigenvalues, psi, occ, orb_pos = aims_read_eigenvalues_and_coefficients(fermi_level, path[:-1], spin=False, debug=False)
#    H, S = aims_read_HS(path[:-1],spin=False)
    KS_out = np.zeros(shape=(20,1*342*1464),dtype='complex128') # hardcoded only for test cases!!!
    KS_out_grp = np.zeros(shape=(20,1,342,1464),dtype='complex128')
    params = []
#    eig_grp = np.zeros(shape=(1,5,10),dtype='complex128')
#    occ_grp = np.zeros(shape=(1,5,10),dtype='complex128')
    for ii in range(N_k_control):
        Ha_to_eV = 1./27.2114
        if reproduce:
            if ii<10:
                num = '00'+str(ii)
            else:
                num = '0'+str(ii)
            n1=path+name+num
            
            if ii in range(nkpts+1)[1:]:
                k_valid = 1
            else:
                k_valid = 0
                
            with FortranFile(n1) as f:
                n_k_points = np.shape(kpoint_weights)[0]
                n_basis = f.read_record('i4')
                n_states_to_save = f.read_record('i4')
                n_spin = f.read_record('i4')
                n_k_points_task = f.read_record('i4')
                params.append([np.array([nkpts],dtype='int32'),n_basis,n_states_to_save,n_spin,np.array([n_k_points],dtype='int32')])
                #print n_k_points_task
                KS_vec=[]
                occ_numbers = []
                KS_eigen = []
                KS_eigen_occ = []
                n_task_list = []
                
                if k_valid:
                    #print(n_spin[0]*n_states_to_save[0]*n_basis[0])
                    #print(n_spin[0],n_states_to_save[0],n_basis[0])
                    for i in range(n_spin[0]*n_states_to_save[0]*n_basis[0]):
                        e1 = f.read_record('f8')
                        #print(e1)
                        e2 = e1[0]+1j*e1[1]
                        if np.abs(e1[0])<1e-10:
                            KS_vec.append(e2)#*0.0) one could introduce a threshold
                        else:
                            KS_vec.append(e2)
                    #print(KS_vec)
                    KS_out[ii-1] = np.array(KS_vec) #np.array(KS_vec).reshape(n_basis[0],n_states_to_save[0], n_spin[0])
                    KS_out_grp[ii-1] = np.array(KS_vec).reshape(n_spin[0],n_states_to_save[0],n_basis[0])
                    for j in range(n_spin[0]*n_states_to_save[0]*n_k_points):
                        e3 = f.read_record('f8')
                        occ_numbers.append(np.array(e3[1]))
                        KS_eigen.append(np.array(e3[0]))
                        KS_eigen_occ.append(e3)
                        
                    eig_grp = np.array(KS_eigen).reshape(n_k_points,n_spin[0],n_states_to_save[0])
                    occ_grp = np.array(occ_numbers).reshape(n_k_points,n_spin[0],n_states_to_save[0])
                else:
                    #print(n_spin[0]*n_states_to_save[0]*n_basis[0])
                    #print(n_spin[0],n_states_to_save[0],n_basis[0])
                    for j in range(n_spin[0]*n_states_to_save[0]*n_k_points):
                        #print(n_spin,n_states_to_save,n_basis,n_k_points)
                        e3 = f.read_record('f8')
                        occ_numbers.append(np.array(e3[1]))
                        KS_eigen.append(np.array(e3[0]))
                        KS_eigen_occ.append(e3)
                    print("Reading header files")
                

    return KS_out, KS_eigen, occ_numbers, KS_out_grp, eig_grp, occ_grp, np.array(params)

def REPRODUCE_RESTART_FILES(path,KS_grp,KSe_grp,occ_grp,N_k_control,params,reproduce=1):
    
    for ii in range(N_k_control):
        Ha_to_eV = 1./27.2114
        nkpts = params[ii,0]#len(kpoint_weights)
        n_k_points = params[ii,4]
        n_basis = params[ii,1]
        n_states_to_save = params[ii,2]
        n_spin = params[ii,3]
        
        if reproduce:
            if ii<10:
                num = '00'+str(ii)
            else:
                num = '0'+str(ii)
            n2=path+'restart'+num
            
            f = FortranFile(n2, 'w')
            f.close()
            
            if ii in range(nkpts[0]+1)[1:]:
                k_valid = 1
            else:
                k_valid = 0
            
            occ = occ_grp[:nkpts[0],:n_spin[0],:n_states_to_save[0]].reshape(nkpts*n_spin*n_states_to_save)
            eigen = KSe_grp[:nkpts[0],:n_spin[0],:n_states_to_save[0]].reshape(nkpts*n_spin*n_states_to_save)
            
            with FortranFile(n2,mode='w') as f:
                f.write_record(n_basis)#n_basis
                f.write_record(n_states_to_save)#n_states_to_save
                f.write_record(n_spin)#n_spin

                if k_valid:
                    n_k_points_task = np.array([1],dtype='int32')
                    f.write_record(n_k_points_task)#######
                    vec = KS_grp[ii-1].reshape(n_spin*n_states_to_save*n_basis)
                    for i in range(n_spin[0]*n_states_to_save[0]*n_basis[0]):
                        ee = vec[i]#np.array([vec[i].real,vec[i].imag],dtype='float64')
                        f.write_record(ee)
                    for j in range(n_spin[0]*n_states_to_save[0]*n_k_points[0]):
                        ee1 = occ[j]
                        ee2 = eigen[j]
                        eee = np.array([ee2,ee1],dtype='float64')
                        f.write_record(eee)
                        #f.write_record(occ_numbers[j])
                else:
                    n_k_points_task = np.array([0],dtype='int32')
                    f.write_record(n_k_points_task)#######
                    for j in range(n_spin[0]*n_states_to_save[0]*n_k_points[0]):
                        ee1 = occ[j]
                        ee2 = eigen[j]
                        eee = np.array([ee2,ee1],dtype='float64')
                        f.write_record(eee)
                    print("Writing header files")


def WRITE_STANDARD_FILE(path,KS_grp,KSe_grp,occ_grp,N_k_control,params,reproduce=1):
    for ii in range(N_k_control):
        Ha_to_eV = 1./27.2114
        nkpts = params[ii,0]#len(kpoint_weights)
        n_k_points = params[ii,4]
        n_basis = params[ii,1]
        n_states_to_save = params[ii,2]
        n_spin = params[ii,3]
        
        if reproduce:
            
            if ii in range(nkpts[0]+1)[1:]:
                k_valid = 1
            else:
                k_valid = 0
            
            occ = occ_grp[:nkpts[0],:n_spin[0],:n_states_to_save[0]].reshape(nkpts*n_spin*n_states_to_save)
            eigen = KSe_grp[:nkpts[0],:n_spin[0],:n_states_to_save[0]].reshape(nkpts*n_spin*n_states_to_save)
            
            with open('CAB_constructed.dat', 'w+') as f:
                f.write_record(nkpts)
                f.write_record(n_spin)
                f.write_record(n_states_to_save)
                f.write_record(n_basis)
                
                if k_valid:
                    vec = KS_grp[ii-1].reshape(n_spin*n_states_to_save*n_basis)
                    for i in range(n_spin[0]*n_states_to_save[0]*n_basis[0]):
                        ee = vec[i]#np.array([vec[i].real,vec[i].imag],dtype='float64')
                        f.write_record(ee)
            

#####################################################################################################
#####################################################################################################                    
def write_restart_files_from_input(path,N_k_control,eigenvalues,psi,occ,kpoint_weights,reproduce=0):
    #atoms = read(path+'geometry.in')
    #cell = atoms.cell
    #filename = path+'output.aimsrestart'
    #fermi_level, kpoint_weights = aims_read_fermi_and_kpoints(filename, cell)
    nkpts = len(kpoint_weights)
    for ii in range(N_k_control):
        Ha_to_eV = 1./27.2114
        if ii<10:
            num = '00'+str(ii)
        else:
            num = '0'+str(ii)
        n2='xtest'+num
        
        f = FortranFile(path+n2, 'w')
        f.close()
        
        if ii in range(nkpts+1)[1:]:
            k_valid = 1
        else:
            k_valid = 0
        
        with FortranFile(path+n2,mode='w') as f:
            n_k_points = np.shape(kpoint_weights)[0]
            n_basis = np.array([psi.shape[3]],dtype='int32')
            n_states_to_save = np.array([occ.shape[2]],dtype='int32')
            n_spin = np.array([occ.shape[1]],dtype='int32')
            #n_k_points_task = np.array([1],dtype='int32') #hardcoded but could be read in from the output
            f.write_record(n_basis)
            f.write_record(n_states_to_save)
            f.write_record(n_spin)
            ll=0
            if k_valid:
                n_k_points_task = np.array([1],dtype='int32')
                f.write_record(n_k_points_task)#######
                for i_k in range(n_k_points_task[0]):#n_k_points_task
                    for i_spin in range(n_spin[0]):
                        for i_states in range(n_states_to_save[0]):
                            for i_basis in range(n_basis[0]):
                                ee1 = psi[ii-1,i_spin,i_states,i_basis]#write the eigenvectors for the respective k-point to the file ii-1
                                ee = np.array([ee1.real,ee1.imag],dtype='float64')
                                ll+=1
                                f.write_record(ee)
                                
                for i_k in range(n_k_points):
                    for i_spin in range(n_spin[0]):
                        for i_states in range(n_states_to_save[0]):
                            ee2 = eigenvalues[i_k,i_spin,i_states]*Ha_to_eV
                            ee3 = occ[i_k,i_spin,i_states]
                            ee = np.array([ee2,ee3],dtype='float64')
                            f.write_record(ee)
            else:
                n_k_points_task = np.array([0],dtype='int32')
                f.write_record(n_k_points_task[0])#######
                for i_k in range(n_k_points):
                    for i_spin in range(n_spin[0]):
                        for i_states in range(n_states_to_save[0]):
                            ee2 = eigenvalues[i_k,i_spin,i_states]*Ha_to_eV
                            ee3 = occ[i_k,i_spin,i_states]
                            ee = np.array([ee2,ee3],dtype='float64')
                            f.write_record(ee)
                print("Printing header files")

def read_restart_files(path,N_k_control,reproduce=1):
    atoms = read(path+'geometry.in')
    cell = atoms.cell
    filename = path+'output.aimsrestart'
    fermi_level, kpoint_weights = aims_read_fermi_and_kpoints(filename, cell)
    nkpts = len(kpoint_weights)
    f=FortranFile(path+'test001','r')
    n_basis = f.read_record('i4')
    n_states_to_save = f.read_record('i4')
    n_spin = f.read_record('i4')
    n_k_points_task = f.read_record('i4')
    f.close()
    KS_vec=np.zeros(shape=(N_k_control,n_spin[0],n_states_to_save[0],n_basis[0]),dtype='complex128',order='F')
    KS_eigen=np.zeros(shape=(N_k_control,n_spin[0],n_states_to_save[0]),dtype='float64',order='F')
    occ_numbers=np.zeros(shape=(N_k_control,n_spin[0],n_states_to_save[0]),dtype='float64',order='F')
    for ii in range(N_k_control):
        Ha_to_eV = 1./27.2114
        if reproduce:
            if ii<10:
                num = '00'+str(ii)
            else:
                num = '0'+str(ii)
            n1='test'+num
            #n2='test'+num
            
            #f = FortranFile(n2, 'w')
            #f.close()
            
            if ii in range(nkpts+1)[1:]:
                k_valid = 1
            else:
                k_valid = 0
                
            with FortranFile(path+n1,'r') as f:
                #my_kpoint_file=f.read_record('i4')
                n_basis = f.read_record('i4')
                n_states_to_save = f.read_record('i4')
                n_spin = f.read_record('i4')
                n_k_points_task = f.read_record('i4')
                #print(n_basis,n_states_to_save,n_spin)
                if k_valid:
                    for i in range(n_spin[0]):
                        for j in range(n_states_to_save[0]):
                            for k in range(n_basis[0]):
                                e2=f.read_record('f8')
                                #print(e2)
                                KS_vec[ii,i,j,k]=np.array(e2[0]+1j*e2[1])
                    
                    for i in range(n_spin[0]):
                        for j in range(n_states_to_save[0]):
                            e3 = f.read_record('f8')
                            KS_eigen[ii,i,j] = np.array(e3[0])
                            occ_numbers[ii,i,j] = np.array(e3[1])
                    
                else:
                    for i in range(n_spin[0]):
                        for j in range(n_states_to_save[0]):
                            e3 = f.read_record('f8')
                            KS_eigen[ii,i,j] = np.array(e3[0])
                            occ_numbers[ii,i,j] = np.array(e3[1])
            f.close()
    return np.array(KS_vec),np.array(KS_eigen),  np.array(occ_numbers)

def write_restart_files2(path,N_k_control,eigenvalues,psi,occ,kpoint_weights):
    nkpts = len(kpoint_weights)
    for ii in range(N_k_control):
        Ha_to_eV = 1./27.2114
        if ii<10:
            num = '00'+str(ii)
        else:
            num = '0'+str(ii)
        n2='Test'+num
        
        f = FortranFile(path+n2, 'w')
        f.close()
        
        if ii in range(nkpts+1)[1:]:
            k_valid = 1
        else:
            k_valid = 0
        
        with FortranFile(path+n2,mode='w') as f:
            n_k_points = np.shape(kpoint_weights)[0]
            n_basis = np.array([psi.shape[0]],dtype='int32')
            n_states_to_save = np.array([occ.shape[0]],dtype='int32')
            n_spin = np.array([occ.shape[1]],dtype='int32')
            #n_k_points_task = np.array([1],dtype='int32') #hardcoded but could be read in from the output
            f.write_record(n_basis)
            f.write_record(n_states_to_save)
            f.write_record(n_spin)
            ll=0
            if k_valid:
                n_k_points_task = np.array([1],dtype='int32')
                f.write_record(n_k_points_task)#######
                for i_k in range(n_k_points_task[0]):#n_k_points_task
                    for i_spin in range(n_spin[0]):
                        for i_states in range(n_states_to_save[0]):
                            for i_basis in range(n_basis[0]):
                                ee1 = psi[i_basis,i_states,i_spin,ii]#write the eigenvectors for the respective k-point to the file ii-1
                                ee = np.array([ee1],dtype='float64')
                                ll+=1
                                f.write_record(ee)
                                
                for i_k in range(n_k_points):
                    for i_spin in range(n_spin[0]):
                        for i_states in range(n_states_to_save[0]):
                            ee2 = eigenvalues[i_states,i_spin,i_k]#*Ha_to_eV
                            ee3 = occ[i_states,i_spin,i_k]
                            ee = np.array([ee2,ee3],dtype='float64')
                            f.write_record(ee)
            else:
                n_k_points_task = np.array([0],dtype='int32')
                f.write_record(n_k_points_task)#######
                for i_k in range(n_k_points):
                    for i_spin in range(n_spin[0]):
                        for i_states in range(n_states_to_save[0]):
                            ee2 = eigenvalues[i_states,i_spin,i_k]#*Ha_to_eV
                            ee3 = occ[i_states,i_spin,i_k]
                            #print(ee2,ee3)
                            ee = np.array([ee2,ee3],dtype='float64')
                            f.write_record(ee)
                print("Writing header files")
                
def read_restart_files_TEST(path,N_k_control,reproduce=1,name='test'):
    atoms = read(path+'geometry.in')
    cell = atoms.cell
    filename = path+'output.aimsrestart'
    fermi_level, kpoint_weights = aims_read_fermi_and_kpoints(filename, cell)
    nkpts = len(kpoint_weights)
    f=FortranFile(path+name+'001','r')
    n_basis = f.read_record('i4')
    n_states_to_save = f.read_record('i4')
    n_spin = f.read_record('i4')
    n_k_points_task = f.read_record('i4')
    f.close()
    KS_vec=np.zeros(shape=(n_basis[0],n_states_to_save[0],n_spin[0],N_k_control),dtype='float64',order='F')
    KS_eigen=np.zeros(shape=(n_states_to_save[0],n_spin[0],N_k_control),dtype='float64',order='F')
    occ_numbers=np.zeros(shape=(n_states_to_save[0],n_spin[0],N_k_control),dtype='float64',order='F')
    for ii in range(N_k_control):
        Ha_to_eV = 1./27.2114
        if reproduce:
            if ii<10:
                num = '00'+str(ii)
            else:
                num = '0'+str(ii)
            n1=name+num
            #n2='test'+num
            
            #f = FortranFile(n2, 'w')
            #f.close()
            
            if ii in range(nkpts+1)[1:]:
                k_valid = 1
            else:
                k_valid = 0
                
            with FortranFile(path+n1,'r') as f:
                #my_kpoint_file=f.read_record('i4')
                n_basis = f.read_record('i4')
                n_states_to_save = f.read_record('i4')
                n_spin = f.read_record('i4')
                n_k_points_task = f.read_record('i4')
                print(n_basis,n_states_to_save,n_spin)
                if k_valid:
                    for i in range(n_spin[0]):
                        for j in range(n_states_to_save[0]):
                            for k in range(n_basis[0]):
                                e2=f.read_record('f8')
                                #print(e2)
                                KS_vec[k,j,i,ii]=np.array(e2[0])
                    for i_k in range(nkpts):
                        for i in range(n_spin[0]):
                            for j in range(n_states_to_save[0]):
                                e3 = f.read_record('f8')
                                KS_eigen[j,i,i_k] = np.array(e3[0])
                                occ_numbers[j,i,i_k] = np.array(e3[1])
                    
                else:
                    for i_k in range(nkpts):
                        for i in range(n_spin[0]):
                            for j in range(n_states_to_save[0]):
                                e3 = f.read_record('f8')
                                KS_eigen[j,i,i_k] = np.array(e3[0])
                                occ_numbers[j,i,i_k] = np.array(e3[1])
    return np.array(KS_vec),np.array(KS_eigen),  np.array(occ_numbers)

#def read_restart(path,outfile):
#    """
#    Read a FHIaims restart file in binary format.
#
#    Parameters
#    ----------
#    outfile : str
#        FHIaims output file.
#
#    Returns
#    -------
#    KS_eigenvector :
#        The KS eigenvector
#    KS_eigenvalue :
#        The KS eigenvalues
#    occupations :
#        The occupations.
#    """
#    base, aimsout = os.path.split(outfile)
#    meta = parse_aimsout(path+outfile)
#
#    if base != "":
#        meta["restartfile"] = os.path.join(base, meta["restartfile"])
#
#    if meta["periodic"]:
#        ks_ev, ks_e, occ = \
#            restartutils.read_restart_periodic(meta["n_states_saved"],
#                                               meta["n_basis"],
#                                               meta["n_spin"],
#                                               meta["restartfile"])
#
#        KS_eigenvector = ks_ev.reshape(meta["n_basis"],
#                                       meta["n_states_saved"],
#                                       meta["n_spin"])
#
#    else:
#        ks_ev, ks_e, occ = \
#            restartutils.read_restart_cluster(meta["n_states_saved"],
#                                              meta["n_basis"],
#                                              meta["n_spin"],
#                                              meta["restartfile"])
#
#        KS_eigenvector = ks_ev.reshape(meta["n_basis"],
#                                       meta["n_states_saved"],
#                                       meta["n_spin"])
#
#    KS_eigenvalue = list()
#    for item in ks_e:
#        KS_eigenvalue.append([float(x) for x in item])
#    KS_eigenvalue = np.array(KS_eigenvalue)
#
#    occupations = list()
#    for item in occ:
#        occupations.append([float(x) for x in item])
#    occupations = np.array(occupations)
#
#    return KS_eigenvector, KS_eigenvalue, occupations
#
#def read_restart_TEST(path,meta,name='test'):
#    """
#    Read a FHIaims restart file in binary format.
#
#    Parameters
#    ----------
#    outfile : str
#        FHIaims output file.
#
#    Returns
#    -------
#    KS_eigenvector :
#        The KS eigenvector
#    KS_eigenvalue :
#        The KS eigenvalues
#    occupations :
#        The occupations.
#    """
#    N_k_control = 16 #hardcoded for test purposes.... read them in later!!
#    n_states_saved = meta["n_states"]
#    n_basis = meta["n_basis"]
#    n_spin = meta["n_spin"]
#    periodic = meta["periodic"]
#    nkpts=12 #hardcoded for now
#    KS_ev = []
#    KS_eigenvalue = []
#    occupations = []
#    for ii in range(N_k_control):
#        
#        if ii<10:
#            num = '00'+str(ii)
#        else:
#            num = '0'+str(ii)
#        restartfile=name+num
#        
#        if ii in range(nkpts+1)[1:]:
#            k_valid = 1
#        else:
#            k_valid = 0
#            
#        if k_valid:
#            if periodic:
#                ks_ev, ks_e, occ = \
#                    restartutils.read_restart_periodic(n_states_saved,
#                                                       n_basis,
#                                                       n_spin,
#                                                       path+restartfile)
#        
#                KS_eigenvector = ks_ev.reshape(n_basis,
#                                               n_states_saved,
#                                               n_spin)
#                KS_ev.append(KS_eigenvector)
##                KS_eigenvalue.append(ks_e)
##                occupations.append(occ)
#        
#
#    KS_eigenvalue = list()
#    for item in ks_e:
#        KS_eigenvalue.append([float(x) for x in item])
#    KS_eigenvalue = np.array(KS_eigenvalue)
#
#    occupations = list()
#    for item in occ:
#        occupations.append([float(x) for x in item])
#    occupations = np.array(occupations)
#
#    return np.array(KS_ev), KS_eigenvalue, occupations
#
#def write_restart(path,filename,KS_eigenvector, KS_eigenvalue, occupations):
#    """
#    Write a FHIaims restart file in binary format.
#
#    Parameters
#    ----------
#    filename : str
#        The filename for the restart file.
#    periodic : bool
#        If True, calculation is with periodic boundary conditions
#    KS_eigenvector : np.array
#        The KS_eigenvector of the system
#    KS_eigenvalue : np.array
#        The KS_eigenvalues of the system
#    occupations : np.array
#        The occupations of the system
#    """
#    N_k_control = 16 #hardcoded for test purposes.... read them in later!!
#    nkpts=12 #haŕdcoded for now
#    periodic = True
#    n_states = KS_eigenvector.shape[1]
#    n_basis = KS_eigenvector.shape[0]
#    n_spin = KS_eigenvector.shape[2]
#    
#    for n_k in range(N_k_control):
#        
#        if n_k<10:
#            num = '00'+str(n_k)
#        else:
#            num = '0'+str(n_k)
#        restartfile=filename+num
#        
#        if n_k in range(nkpts+1)[1:]:
#            k_valid = 1
#        else:
#            k_valid = 0
#            
#        if k_valid:        
#        
#            if periodic:
#                # TODO careful, hack to work with gamma periodic, need to
#                #     change that to proper n_k loop for Martin ;-)
#                #KS_eigenvector = KS_eigenvector.reshape(n_basis, n_states, n_spin, n_k)
#                filename = filename
#                KS_eig = KS_eigenvector[:,:,n_spin-1,n_k-1]
#                restartutils.write_restart_periodic(path+restartfile,
#                                                    KS_eig,
#                                                    KS_eigenvalue,
#                                                    occupations,n_basis,n_states,n_spin)
#        else:
#            
#            #KS_eigenvector = KS_eigenvector.reshape(n_basis, n_states, n_spin, n_k)
#            filename = filename
#            restartutils.write_restart_periodic_eigval(path+restartfile,KS_eigenvalue,occupations,n_basis,n_states,n_spin)
#
#
#import re
#
#def parse_aimsout(outfile):
#    """
#    Parse the aims output file for some information.
#
#    Parameters
#    ----------
#    outfile : str
#        The path to the AIMS output file.
#
#    Returns
#    -------
#    meta : dict
#        Dictionary with restartfile, n_spin, n_empty, n_states, n_basis.
#    """
#    with open(outfile) as f:
#        out = " ".join(f.readlines())
#    regex = {"restartfile":
#             re.compile(r"Writing periodic restart information to file (\S+)"
#                        "| Writing cluster restart information to file (\S+)"),
#             "periodic":
#             re.compile(r"Found k-point grid: \s+ (\d \s+ \d \s+ \d)"),
#             "n_spin":
#             re.compile(r"Number of spin channels \s* : \s* ([12]?)"),
##             "n_empty":
##             re.compile(r"Total number of empty states used during \D* (\d+)"
##                        "| Number of empty states per atom: \D* (\d*)"),
#             "n_states":
#             re.compile(r"Number of Kohn-Sham states \D* (\d*)"),
#             "n_states_saved":
#             re.compile(r"Restart file: Writing \D* (\d*)"),
#             "n_basis":
#             re.compile(r"Maximum number of basis functions \D* (\d*)"),
#             "proc_gammaK":
#             re.compile(r"\| K-points in task\s+(\d):\s+1")}
##             "N_k_control":
##             re.compile(r"| Number of k-points [\s:] \d ")}
#
#    meta = dict()
#    # Maybe the whole loop should be written explicetly to avoid too many
#    # try-except clauses...
#    for key, value in regex.items():
#        if key == "periodic":
#            meta["periodic"] = bool(value.findall(out))
##        elif key == "n_empty":
##            meta["n_empty"] = int((x for x in value.findall(out)[0]
##                                   if x != "").next())
##        elif key == "N_k_control":
##            print([x for x in value.findall(out)[0] if x != " "][0])
##            meta["N_k_control"] = int((x for x in value.findall(out)[0]
##                                   if x != " ").__next__()) 
#            
#        elif key == "restartfile":
#            meta[key] = [x for x in value.findall(out)[0] if x != " "][0]
#        else:
#            try:
#                meta[key] = int(value.findall(out)[0])
#            except ValueError:
#                meta[key] = value.findall(out)[0]
#            except TypeError:
#                pass
#            except IndexError:
#                # regex found nothing, e.g. periodic in cluster v.v.
#                pass
#
#    #meta["n_states_saved"] = min(meta["n_states"],
#                                 #meta["n_states"] - meta["n_empty"] + 3)
#
#    if meta["periodic"] is True:
#        # assumes aims does 3 numbers for restarts...
#        stub = meta["restartfile"][:-3]
#        meta["restartfile"] = stub + "{0:03d}".format(meta["proc_gammaK"])
#
#    return meta


        