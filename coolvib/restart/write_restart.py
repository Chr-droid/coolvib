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
    
#--------------------------------------------------------------------------------------------------------
    


N_k_control=16
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
    for ii in range(N_k_control):
        Ha_to_eV = 1./27.2114
        if reproduce:
            if ii<10:
                num = '00'+str(ii)
            else:
                num = '0'+str(ii)
            n1='test'+num
            n2='test'+num
            
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
                #n_task_list.append(n_k_points_task)
                if k_valid:
                    for i in range(n_spin[0]*n_states_to_save[0]*n_basis[0]):
                        e2 = f.read_record('f8')
                        if np.abs(e2[0])<1e-10:
                            KS_vec.append(e2)#*0.0) one could introduce a threshold
                        else:
                            KS_vec.append(e2)
                    for j in range(n_spin[0]*n_states_to_save[0]*n_k_points):
                        e3 = f.read_record('f8')
                        occ_numbers.append(np.array(e3[1]))
                        KS_eigen.append(np.array(e3[0]))
                        KS_eigen_occ.append(e3)
                else:
                    for j in range(n_spin[0]*n_states_to_save[0]*n_k_points):
                        e3 = f.read_record('f8')
                        occ_numbers.append(np.array(e3[1]))
                        KS_eigen.append(np.array(e3[0]))
                        KS_eigen_occ.append(e3)
                    print("Printing header files")
            
                
                    
            with FortranFile(n2,mode='w') as f:
                f.write_record(n_basis)
                f.write_record(n_states_to_save)
                f.write_record(n_spin)
                f.write_record(n_k_points_task)
                if k_valid:
                    for i in range(n_spin[0]*n_states_to_save[0]*n_basis[0]):
                        f.write_record(KS_vec[i])
                    for j in range(n_spin[0]*n_states_to_save[0]*n_k_points):
                        f.write_record(KS_eigen_occ[j])
                        #f.write_record(occ_numbers[j])
                else:
                    for j in range(n_spin[0]*n_states_to_save[0]*n_k_points):
                        f.write_record(KS_eigen_occ[j])
                    print("Printing header files")
        else:
            if ii<10:
                num = '00'+str(ii)
            else:
                num = '0'+str(ii)
            n1='test'+num
            n2='test'+num
            
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
                    
def write_restart_files_from_input(path,N_k_control,eigenvalues,psi,occ,orb_pos,kpoint_weights,reproduce=0):
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
        n2='test'+num
        
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
                    
#write_restart_files(path,N_k_control,reproduce)
        