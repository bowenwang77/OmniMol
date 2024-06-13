import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import os
from scipy.spatial.distance import pdist, squareform

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from io import BytesIO
from PIL import Image
from rdkit.Geometry import Point2D
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def calculate_energy(smiles, atom_positions, atom_numbers):
    # # Example data - replace with your data
    # smiles = 'CN(C)CCCN1c2ccccc2CCc2ccccc21'  # SMILES string
    # position_relax=position_init.copy()+np.array(sample["targets"]["deltapos"][idx].cpu())[real_mask]
    # atom_positions = position_relax  # Replace with your atomic positions (45, 3) array
    # atom_numbers = atom  # Replace with your atomic numbers (45,) array

    # Step 1: Create a molecule from the SMILES string
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # Step 2: Set the 3D coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, pos in enumerate(atom_positions):
        if i < mol.GetNumAtoms():  # Check to prevent index out of range
            conf.SetAtomPosition(i, np.float64(pos))
    mol.AddConformer(conf, assignId=True)

    # Step 3: Apply the MMFF
    AllChem.MMFFSanitizeMolecule(mol)
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
    force_field = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=0)

    energy = force_field.CalcEnergy()
    # print("Molecular Energy (MMFF):", energy)
    return energy




def visualize_trace(pc_trace_delta,pos,label_delta,non_padding_mask,directory="visualize/default"):
    fig = plt.figure(figsize=(50,50))
    num_trace = pc_trace_delta.shape[0]
    num_frame = pc_trace_delta.shape[1]

    for trace_id in range(num_trace):
        mins = pos[trace_id][non_padding_mask[trace_id]].min(axis=0)
        maxs = pos[trace_id][non_padding_mask[trace_id]].max(axis=0)
        spans = maxs-mins
        trace_inst = pc_trace_delta[trace_id]
        for frame_id in range(num_frame):
            # pc = pc_trace_delta[trace_id,frame_id]+pos[trace_id]
            pc = pc_trace_delta[trace_id,frame_id][non_padding_mask[trace_id]]+pos[trace_id][non_padding_mask[trace_id]]
            x = pc[:, 0]
            y = pc[:, 1]
            z = pc[:, 2]
            pc_l = label_delta[trace_id][non_padding_mask[trace_id]]+pos[trace_id][non_padding_mask[trace_id]]
            x_l = pc_l[:, 0]
            y_l = pc_l[:, 1]
            z_l = pc_l[:, 2]
            ax = fig.add_subplot(num_trace,num_frame, trace_id*num_frame+(frame_id+1), projection ='3d')
            ax.scatter(x,y,z,marker='o',c='b')
            ax.scatter(x_l,y_l,z_l,marker='x',c='r')
            ax.set_xlim([mins[0]-0.1*spans[0],maxs[0]+0.1*spans[0]])
            ax.set_ylim([mins[1]-0.1*spans[1],maxs[1]+0.1*spans[1]])
            ax.set_zlim([mins[2]-0.1*spans[2],maxs[2]+0.1*spans[2]])
            label_drift_sum = np.linalg.norm(label_delta[trace_id],axis=1).sum()
            pred_drift_sum = np.linalg.norm(label_delta[trace_id]-pc_trace_delta[trace_id,frame_id],axis=1).sum()
            print(trace_id,frame_id,pred_drift_sum/label_drift_sum)
            ax.set_title("Pos remains:"+str(pred_drift_sum/label_drift_sum))
    plt.savefig(directory)

    fig = plt.figure(figsize=(50,50))
    num_trace = pc_trace_delta.shape[0]
    num_frame = pc_trace_delta.shape[1]

    for trace_id in range(num_trace):
        trace_inst = pc_trace_delta[trace_id]
        all_delta_pos = np.concatenate([pc_trace_delta[trace_id].reshape(-1,3),label_delta[trace_id].reshape(-1,3)],axis=0)
        mins = all_delta_pos.min(axis=0)
        maxs = all_delta_pos.max(axis=0)
        spans = maxs-mins
        for frame_id in range(num_frame):
            # pc = pc_trace_delta[trace_id,frame_id]+pos[trace_id]
            pc = pc_trace_delta[trace_id,frame_id][non_padding_mask[trace_id]]
            x = pc[:, 0]
            y = pc[:, 1]
            z = pc[:, 2]
            pc_l = label_delta[trace_id][non_padding_mask[trace_id]]
            x_l = pc_l[:, 0]
            y_l = pc_l[:, 1]
            z_l = pc_l[:, 2]
            ax = fig.add_subplot(num_trace,num_frame, trace_id*num_frame+(frame_id+1), projection ='3d')
            ax.scatter(x,y,z,marker='o',c='b')
            ax.scatter(x_l,y_l,z_l,marker='x',c='r')
            ax.set_xlim([mins[0]-0.1*spans[0],maxs[0]+0.1*spans[0]])
            ax.set_ylim([mins[1]-0.1*spans[1],maxs[1]+0.1*spans[1]])
            ax.set_zlim([mins[2]-0.1*spans[2],maxs[2]+0.1*spans[2]])
            label_drift_sum = np.linalg.norm(label_delta[trace_id],axis=1).sum()
            pred_drift_sum = np.linalg.norm(label_delta[trace_id]-pc_trace_delta[trace_id,frame_id],axis=1).sum()
            print(trace_id,frame_id,pred_drift_sum/label_drift_sum)
            ax.set_title("Pos remains:"+str(pred_drift_sum/label_drift_sum))
    plt.savefig(directory+"only_drift")

def visualize_trace_poscar(sample, cell,sid, output_vals,e_mean, e_std, d_mean, d_std, directory="visualize/default"):
    import pymatgen
    import numpy as np
    atom_list = [
        1,
        5,
        6,
        7,
        8,
        11,
        13,
        14,
        15,
        16,
        17,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        55,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
    ]
    symbols = ['H','He','Li','Be','B','C','N','O','F','Ne',
            'Na','Mg','Al','Si','P','S','Cl','Ar','K', 'Ca',
            'Sc', 'Ti', 'V','Cr', 'Mn', 'Fe', 'Co', 'Ni',
            'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
            'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
            'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
            'I', 'Xe','Cs', 'Ba','La', 'Ce', 'Pr', 'Nd', 'Pm',
            'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
            'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
            'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
            'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am',
            'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
            'Rf', 'Db', 'Sg', 'Bh','Hs', 'Mt', 'Ds', 'Rg', 'Cn',
            'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']


    atom_list=np.array(atom_list)
    node_output=output_vals['node_output']
    deltapos_trace = output_vals['deltapos_trace']
    num_trace= deltapos_trace.shape[0]
    num_frames = deltapos_trace.shape[1]
    directory_root = directory
    for idx in range(num_trace):
        lattice = np.array(cell[idx].cpu())
        lattice = pymatgen.core.lattice.Lattice(lattice)
        real_mask = sample["net_input"]['real_mask'][idx].cpu()
        tag=np.array(sample["net_input"]['tags'][idx].cpu())[real_mask]
        atom=np.array(sample["net_input"]['atoms'][idx].cpu())[real_mask]
        atom=atom_list[atom-1]
        atom_mark_fix = atom.copy()
        atom_mark_fix[tag==0]=1
        directory=directory_root+"/POSCAR"+str(sid[idx].item())
        if directory.split("/")[2]=="SAA": 
            directory = directory+"_"+symbols[atom[-3]-1]
        if not os.path.exists(directory):
            os.makedirs(directory)
        log=open(directory+"/log.txt","w+")

        position_init=np.array(sample["net_input"]['pos'][idx].cpu())[real_mask]
        ins = pymatgen.core.structure.IStructure(lattice,atom,position_init,coords_are_cartesian=True)
        Poscar = pymatgen.io.vasp.Poscar(ins)
        Poscar.write_file(directory+"/POSCAR_init")
        ins = pymatgen.core.structure.IStructure(lattice,atom_mark_fix,position_init,coords_are_cartesian=True)
        Poscar = pymatgen.io.vasp.Poscar(ins)
        Poscar.write_file(directory+"/POSCAR_init_mark_fix")        

        position_relax=position_init.copy()+np.array(sample["targets"]["deltapos"][idx].cpu())[real_mask]
        ins = pymatgen.core.structure.IStructure(lattice,atom,position_relax,coords_are_cartesian=True)
        Poscar = pymatgen.io.vasp.Poscar(ins)
        Poscar.write_file(directory+"/POSCAR_relaxed")
        ins = pymatgen.core.structure.IStructure(lattice,atom_mark_fix,position_relax,coords_are_cartesian=True)
        Poscar = pymatgen.io.vasp.Poscar(ins)
        Poscar.write_file(directory+"/POSCAR_relaxed_mark_fix")        

        for frame_idx in range(num_frames):
            delta_position_pred = np.array((deltapos_trace[idx,frame_idx]*node_output.new_tensor(d_std)+node_output.new_tensor(d_mean)).detach().cpu())[real_mask]
            position_pred=position_init.copy()
            position_pred[tag!=0]+=delta_position_pred[tag!=0]
            ins = pymatgen.core.structure.IStructure(lattice,atom,position_pred,coords_are_cartesian=True)
            Poscar = pymatgen.io.vasp.Poscar(ins)
            Poscar.write_file(directory+"/POSCAR_pred_interframe"+str(frame_idx))
            ins = pymatgen.core.structure.IStructure(lattice,atom_mark_fix,position_pred,coords_are_cartesian=True)
            Poscar = pymatgen.io.vasp.Poscar(ins)
            Poscar.write_file(directory+"/POSCAR_pred_interframe_mark_fix"+str(frame_idx))        

            label_remain_norm = np.linalg.norm(sample["targets"]["deltapos"][idx][real_mask][tag!=0].detach().cpu(),axis=1)
            pred_remain_norm = np.linalg.norm(sample["targets"]["deltapos"][idx][real_mask][tag!=0].detach().cpu()-delta_position_pred[tag!=0],axis=1)
            print(sid[idx],frame_idx," Pred remain MAE: ",pred_remain_norm.mean()," Label remain MAE: ",label_remain_norm.mean()," Relative remain ratio: ",pred_remain_norm.mean()/label_remain_norm.mean(),)
            log.writelines([str(sid[idx].item()),str(frame_idx)," Pred remain MAE: ",str(pred_remain_norm.mean())," Label remain MAE: ",str(label_remain_norm.mean())," Relative remain ratio: ",str(pred_remain_norm.mean()/label_remain_norm.mean()),'\n'])

        delta_position_pred=np.array((node_output*node_output.new_tensor(d_std)+node_output.new_tensor(d_mean))[idx].detach().cpu())[real_mask]
        position_pred=position_init.copy()
        position_pred[tag!=0]+=delta_position_pred[tag!=0]
        ins = pymatgen.core.structure.IStructure(lattice,atom,position_pred,coords_are_cartesian=True)
        Poscar = pymatgen.io.vasp.Poscar(ins)
        Poscar.write_file(directory+"/POSCAR_pred_final")
        ins = pymatgen.core.structure.IStructure(lattice,atom_mark_fix,position_pred,coords_are_cartesian=True)
        Poscar = pymatgen.io.vasp.Poscar(ins)
        Poscar.write_file(directory+"/POSCAR_pred_final_mark_fix")        

        label_remain_norm = np.linalg.norm(sample["targets"]["deltapos"][idx][real_mask][tag!=0].detach().cpu(),axis=1)
        pred_remain_norm = np.linalg.norm(sample["targets"]["deltapos"][idx][real_mask][tag!=0].detach().cpu()-delta_position_pred[tag!=0],axis=1)
        print(sid[idx]," final pos pred. "," Pred remain MAE: ",pred_remain_norm.mean()," Label remain MAE: ",label_remain_norm.mean()," Relative remain ratio: ",pred_remain_norm.mean()/label_remain_norm.mean(),)
        log.writelines([str(sid[idx].item()),str(frame_idx)," Pred remain MAE: ",str(pred_remain_norm.mean())," Label remain MAE: ",str(label_remain_norm.mean())," Relative remain ratio: ",str(pred_remain_norm.mean()/label_remain_norm.mean()),'\n'])

        print("Energy label: ",sample['targets']['relaxed_energy'][idx].item(), " Energy prediction: ", output_vals['eng_output'][idx].item()*e_std+e_mean, " Energy Absolute Error: ", abs(sample['targets']['relaxed_energy'][idx].item()-(output_vals['eng_output'][idx].item()*e_std+e_mean)))
        log.writelines(["Energy label: ",str(sample['targets']['relaxed_energy'][idx].item()), " Energy prediction: ", str(output_vals['eng_output'][idx].item()*e_std+e_mean), " Energy Absolute Error: ", str(abs(sample['targets']['relaxed_energy'][idx].item()-(output_vals['eng_output'][idx].item()*e_std+e_mean)))])

        print("visualization file saved in: ", directory)
        log.writelines(["visualization file saved in: ", directory])
        log.close()

        # Visualize
        # import pymatgen
        # lattice = np.array(cell)
        # lattice = pymatgen.core.lattice.Lattice(lattice)
        # ins = pymatgen.core.structure.IStructure(lattice,np.array(atoms),np.array(pos),coords_are_cartesian=True)
        # Poscar = pymatgen.io.vasp.Poscar(ins)
        # Poscar.write_file("POSCAR")


        #vis backup
        # import pymatgen
        # import ase
        # import numpy as np

        # coord = np.array([0.1,0.2,0.3])
        # lattice = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # lattice = pymatgen.core.lattice.Lattice(lattice)
        # ele = pymatgen.core.periodic_table.Element.from_Z(10)
        # site = pymatgen.core.sites.PeriodicSite(ele,coord,lattice)
        # sites = [site,site]
        # mol = pymatgen.core.structure.IStructure.from_sites(sites)
        # Poscar = pymatgen.io.vasp.Poscar(mol)
        # Poscar.write_file("POSCAR")


        ##Vis
        # import pymatgen
        # lattice = np.array(cell)
        # lattice = pymatgen.core.lattice.Lattice(lattice)
        # ins = pymatgen.core.structure.IStructure(lattice,np.array(atoms),np.array(pos),coords_are_cartesian=True)
        # Poscar = pymatgen.io.vasp.Poscar(ins)
        # Poscar.write_file("POSCAR")
        # sites = []
        # for i in range(len(pos)):
        #     coord = np.array(pos[i])
        #     ele = pymatgen.core.periodic_table.Element.from_Z(np.array(atoms[i]))
        #     site = pymatgen.core.sites.PeriodicSite(ele,coord,lattice)
        #     sites.append(site)
        #     pass
        # ins = pymatgen.core.structure.IStructure.from_sites(sites)

def visualize_trace_poscar_molecule(sample, cell,sid, output_vals,e_mean, e_std, d_mean, d_std, directory="visualize/default"):
    import pymatgen
    import numpy as np
    atom_list = [
        1,
        5,
        6,
        7,
        8,
        11,
        13,
        14,
        15,
        16,
        17,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        55,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
    ]
    symbols = ['H','He','Li','Be','B','C','N','O','F','Ne',
            'Na','Mg','Al','Si','P','S','Cl','Ar','K', 'Ca',
            'Sc', 'Ti', 'V','Cr', 'Mn', 'Fe', 'Co', 'Ni',
            'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
            'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
            'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
            'I', 'Xe','Cs', 'Ba','La', 'Ce', 'Pr', 'Nd', 'Pm',
            'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
            'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
            'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
            'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am',
            'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
            'Rf', 'Db', 'Sg', 'Bh','Hs', 'Mt', 'Ds', 'Rg', 'Cn',
            'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

    columns = [ 'Additional_Info','SID', 'State', 'Frame_Index', 'Pred_Remain_MAE', 'Label_Remain_MAE', 'Relative_Remain_Ratio', 'System_Energy']
    atom_list=np.array(atom_list)
    node_output=output_vals['node_output']
    deltapos_trace = output_vals['deltapos_trace']
    num_trace= deltapos_trace.shape[0]
    num_frames = deltapos_trace.shape[1]
    directory_root = directory
    for idx in range(num_trace):
        data = pd.DataFrame(columns=columns)

        lattice = np.eye(3)*20
        lattice = pymatgen.core.lattice.Lattice(lattice)
        real_mask = sample["net_input"]['real_mask'][idx].cpu()
        tag=np.array(sample["net_input"]['tags'][idx].cpu())[real_mask]
        atom=np.array(sample["net_input"]['atoms'][idx].cpu())[real_mask]
        prop_name = sample['task_input']['property_name'][idx]
        atom=atom_list[atom-1]
        smiles = sample["task_input"]['smi'][idx]
        atom_mark_fix = atom.copy()
        atom_mark_fix[tag==0]=1
        directory=directory_root+"/"+smiles[:10]+"_"+str(sid[idx].item())+"_"+prop_name
        if directory.split("/")[2]=="SAA": 
            directory = directory+"_"+symbols[atom[-3]-1]
        if not os.path.exists(directory):
            os.makedirs(directory)
        log=open(directory+"/log.txt","w+")

        position_init=np.array(sample["net_input"]['pos'][idx].cpu())[real_mask]+10
        ins = pymatgen.core.structure.IStructure(lattice,atom,position_init,coords_are_cartesian=True)
        energy = calculate_energy(smiles, position_init, atom)
        Poscar = pymatgen.io.vasp.Poscar(ins)
        Poscar.write_file(directory+"/POSCAR_init")
        # ins = pymatgen.core.structure.IStructure(lattice,atom_mark_fix,position_init,coords_are_cartesian=True)
        # Poscar = pymatgen.io.vasp.Poscar(ins)
        # Poscar.write_file(directory+"/POSCAR_init_mark_fix")          
        label_remain_norm = np.linalg.norm(sample["targets"]["deltapos"][idx][real_mask][tag!=0].detach().cpu(),axis=1)
        pred_remain_norm = np.linalg.norm(sample["targets"]["deltapos"][idx][real_mask][tag!=0].detach().cpu(),axis=1)
        print("SID: ", sid[idx].item()," Initial:"," Pred remain MAE: ",pred_remain_norm.mean()," Label remain MAE: ",label_remain_norm.mean()," Relative remain ratio: ",pred_remain_norm.mean()/label_remain_norm.mean(), " Potential Energy:", energy)
        log.writelines([str(sid[idx].item()),"Initial "," Pred remain MAE: ",str(label_remain_norm.mean())," Label remain MAE: ",str(label_remain_norm.mean())," Relative remain ratio: ",str(pred_remain_norm.mean()/label_remain_norm.mean()), " System energy: ",str(energy),'\n'])    
        data = data.append({
            'Additional_Info': smiles+"_"+str(sid[idx].item())+"_"+prop_name,
            'SID': sid[idx].item(),
            'State': 'Initial',
            'Frame_Index': 'N/A',
            'Pred_Remain_MAE': pred_remain_norm.mean(),
            'Label_Remain_MAE': label_remain_norm.mean(),
            'Relative_Remain_Ratio': pred_remain_norm.mean() / label_remain_norm.mean(),
            'System_Energy': energy
        }, ignore_index=True)

        for frame_idx in range(num_frames):
            delta_position_pred = np.array((deltapos_trace[idx,frame_idx]*node_output.new_tensor(d_std)+node_output.new_tensor(d_mean)).detach().cpu())[real_mask]
            position_pred=position_init.copy()
            position_pred[tag!=0]+=delta_position_pred[tag!=0]
            ins = pymatgen.core.structure.IStructure(lattice,atom,position_pred,coords_are_cartesian=True)
            energy = calculate_energy(smiles, position_pred, atom)
            Poscar = pymatgen.io.vasp.Poscar(ins)
            Poscar.write_file(directory+"/POSCAR_pred_interframe"+str(frame_idx))
            # ins = pymatgen.core.structure.IStructure(lattice,atom_mark_fix,position_pred,coords_are_cartesian=True)
            # Poscar = pymatgen.io.vasp.Poscar(ins)
            # Poscar.write_file(directory+"/POSCAR_pred_interframe_mark_fix"+str(frame_idx))        
            pred_remain_norm = np.linalg.norm(sample["targets"]["deltapos"][idx][real_mask][tag!=0].detach().cpu()-delta_position_pred[tag!=0],axis=1)
            print("SID: ", sid[idx].item(), " Interframe: ",frame_idx," Pred remain MAE: ",pred_remain_norm.mean()," Label remain MAE: ",label_remain_norm.mean()," Relative remain ratio: ",pred_remain_norm.mean()/label_remain_norm.mean(), " Potential Energy:", energy)
            log.writelines([str(sid[idx].item()),str(frame_idx)," Pred remain MAE: ",str(pred_remain_norm.mean())," Label remain MAE: ",str(label_remain_norm.mean())," Relative remain ratio: ",str(pred_remain_norm.mean()/label_remain_norm.mean()), " System energy: ",str(energy),'\n'])
            data = data.append({
                'Additional_Info': smiles+"_"+str(sid[idx].item())+"_"+prop_name,
                'SID': sid[idx].item(),
                'State': 'Intermediate',
                'Frame_Index': frame_idx,
                'Pred_Remain_MAE': pred_remain_norm.mean(),
                'Label_Remain_MAE': label_remain_norm.mean(),
                'Relative_Remain_Ratio': pred_remain_norm.mean() / label_remain_norm.mean(),
                'System_Energy': energy
            }, ignore_index=True)

        delta_position_pred=np.array((node_output*node_output.new_tensor(d_std)+node_output.new_tensor(d_mean))[idx].detach().cpu())[real_mask]
        position_pred=position_init.copy()
        position_pred[tag!=0]+=delta_position_pred[tag!=0]
        ins = pymatgen.core.structure.IStructure(lattice,atom,position_pred,coords_are_cartesian=True)
        energy = calculate_energy(smiles, position_pred, atom)
        Poscar = pymatgen.io.vasp.Poscar(ins)
        Poscar.write_file(directory+"/POSCAR_pred_final")
        # ins = pymatgen.core.structure.IStructure(lattice,atom_mark_fix,position_pred,coords_are_cartesian=True)
        # Poscar = pymatgen.io.vasp.Poscar(ins)
        # Poscar.write_file(directory+"/POSCAR_pred_final_mark_fix")        
        label_remain_norm = np.linalg.norm(sample["targets"]["deltapos"][idx][real_mask][tag!=0].detach().cpu(),axis=1)
        pred_remain_norm = np.linalg.norm(sample["targets"]["deltapos"][idx][real_mask][tag!=0].detach().cpu()-delta_position_pred[tag!=0],axis=1)
        print("SID: ", sid[idx].item(), " Final pred: "," Pred remain MAE: ",pred_remain_norm.mean()," Label remain MAE: ",label_remain_norm.mean()," Relative remain ratio: ",pred_remain_norm.mean()/label_remain_norm.mean(), " Potential Energy:", energy)
        log.writelines([str(sid[idx].item()),str(frame_idx)," Pred remain MAE: ",str(pred_remain_norm.mean())," Label remain MAE: ",str(label_remain_norm.mean())," Relative remain ratio: ",str(pred_remain_norm.mean()/label_remain_norm.mean()), " System energy: ",str(energy),'\n'])
        data = data.append({
            'Additional_Info': smiles+"_"+str(sid[idx].item())+"_"+prop_name,
            'SID': sid[idx].item(),
            'State': 'Final Pred',
            'Frame_Index': 'N/A',
            'Pred_Remain_MAE': pred_remain_norm.mean(),
            'Label_Remain_MAE': label_remain_norm.mean(),
            'Relative_Remain_Ratio': pred_remain_norm.mean() / label_remain_norm.mean(),
            'System_Energy': energy
        }, ignore_index=True)


        position_relax=position_init.copy()+np.array(sample["targets"]["deltapos"][idx].cpu())[real_mask]
        ins = pymatgen.core.structure.IStructure(lattice,atom,position_relax,coords_are_cartesian=True)
        energy = calculate_energy(smiles, position_relax, atom)
        Poscar = pymatgen.io.vasp.Poscar(ins)
        Poscar.write_file(directory+"/POSCAR_relaxed")
        # ins = pymatgen.core.structure.IStructure(lattice,atom_mark_fix,position_relax,coords_are_cartesian=True)
        # Poscar = pymatgen.io.vasp.Poscar(ins)
        # Poscar.write_file(directory+"/POSCAR_relaxed_mark_fix")  
        is_reg = sample['net_input']['is_reg'][idx]
        print("SID: ", sid[idx].item()," Relaxed Ground Truth. "," Pred remain MAE: ", str(0) ," Label remain MAE: ",label_remain_norm.mean()," Relative remain ratio: ",pred_remain_norm.mean()/label_remain_norm.mean(), " Potential Energy:", energy)
        log.writelines([str(sid[idx].item()),str(frame_idx)," Pred remain MAE: ", str(0) ," Label remain MAE: ",str(label_remain_norm.mean())," Relative remain ratio: ",str(pred_remain_norm.mean()/label_remain_norm.mean()), " System energy: ",str(energy),'\n'])
        data = data.append({
            'Additional_Info': smiles+"_"+str(sid[idx].item())+"_"+prop_name,
            'SID': sid[idx].item(),
            'State': 'Relaxed',
            'Frame_Index': 'N/A',
            'Pred_Remain_MAE': pred_remain_norm.mean(),
            'Label_Remain_MAE': label_remain_norm.mean(),
            'Relative_Remain_Ratio': pred_remain_norm.mean() / label_remain_norm.mean(),
            'System_Energy': energy
        }, ignore_index=True)


        if is_reg:
            print("Energy label: ",sample['targets']['relaxed_energy'][idx].item(), " Prop prediction: reg", output_vals['reg_output'][idx].item()*e_std+e_mean, " Energy Absolute Error: ", str(abs(sample['targets']['relaxed_energy'][idx].item()-(output_vals['reg_output'][idx].item()*e_std+e_mean))))
            log.writelines(["Energy label: ",str(sample['targets']['relaxed_energy'][idx].item()), " Prop prediction: reg", str(output_vals['reg_output'][idx].item()*e_std+e_mean), " Energy Absolute Error: ", str(abs(sample['targets']['relaxed_energy'][idx].item()-(output_vals['reg_output'][idx].item()*e_std+e_mean)))])
        else:
            print("Energy label: ",sample['targets']['relaxed_energy'][idx].item(), " Prop prediction cls: ", output_vals['cls_output'][idx][-1].item())
            log.writelines(["Energy label: ",str(sample['targets']['relaxed_energy'][idx].item()), " Prop prediction cls: ", str(output_vals['cls_output'][idx][-1].item())])    

        print("\n visualization file saved in: ", directory)
        log.writelines(["\n visualization file saved in: ", directory])
        log.close()
        data.to_csv(directory+"/log.csv",index=False)



def plot_mol_with_attn(atoms_full, smi, attn,save_dir, idx=0, with_hydrogen = True, prop_name = "molecule", pos = None):
    # Load molecule and compute 2D coordinates
    mol = Chem.MolFromSmiles(smi)
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    rdDepictor.Compute2DCoords(mol)

    def attention_to_color(attn_val):
        norm = Normalize(vmin=0, vmax=1)  # Adjusted normalization
        cmap = plt.cm.Reds
        return cmap(norm(attn_val)+0.2)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(20, 10))

    # Draw molecule and get transformed coordinates
    d = rdMolDraw2D.MolDraw2DCairo(400, 400)
    d.DrawMolecule(mol)
    d.FinishDrawing()
    img_data = d.GetDrawingText()
    img = Image.open(BytesIO(img_data))
    ax.imshow(img)
    ax.axis('off')

    # Get transformed coordinates
    if with_hydrogen:
        non_hydrogen_indices = list(np.arange(atoms_full.shape[0]))
    else:
        non_hydrogen_indices = [index for index, a in enumerate(atoms_full) if a != 1]
    coords_2d = [Point2D(mol.GetConformer().GetAtomPosition(int(index)).x, mol.GetConformer().GetAtomPosition(int(index)).y) for index in non_hydrogen_indices]
    coords_transformed = [d.GetDrawCoords(coord) for coord in coords_2d]

    # Adjust the radius using a logarithmic scale
    def adjusted_radius(attn_val):
        base_radius = 14
        return base_radius + 30*attn_val
    hydrogen_lines = []
    # Calculate pairwise distances in the 3D space
    distances = squareform(pdist(pos))
    
    for index, atom_type in enumerate(atoms_full):
        if atom_type == 1:  # If it's a hydrogen atom
            # Exclude hydrogen atoms from the distances
            non_hydrogen_distances = distances[index, atoms_full != 1]
            
            # Get the index of the nearest non-hydrogen atom
            nearest_atom_idx = np.argmin(non_hydrogen_distances)
            
            # Add the attention value of the hydrogen atom to its nearest non-hydrogen atom
            attn[nearest_atom_idx] += attn[index]
            hydrogen_lines.append((index,nearest_atom_idx))

    if not with_hydrogen:
        # Filter out hydrogen atoms for plotting
        # atoms_full = atoms_full[atoms_full != 1]
        attn = attn[atoms_full != 1]
        pos = pos[atoms_full != 1]

    # Draw attention circles and display the 3D positions
    for coord, attn_val, position in zip(coords_transformed, attn, pos):
        if attn_val> 0.02:
            color = attention_to_color(attn_val)
            circle = plt.Circle((coord.x, coord.y), adjusted_radius(attn_val), color=color, alpha=0.6)
            ax.add_patch(circle)

            # Display the 3D position as text next to the atom
            # text_x = coord.x + 15  # Adjust these values as necessary to place the text correctly
            # text_y = coord.y + 15
            # position_text = f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}), {attn_val:.2f}"
            # ax.text(text_x, text_y, position_text, fontsize=5, ha='center', va='center')
    
    if with_hydrogen:
        # Draw green lines for each pair in hydrogen_lines
        for hydrogen_idx, nearest_atom_idx in hydrogen_lines:
            start_coord = d.GetDrawCoords(Point2D(mol.GetConformer().GetAtomPosition(int(hydrogen_idx)).x, mol.GetConformer().GetAtomPosition(int(hydrogen_idx)).y))
            end_coord = d.GetDrawCoords(Point2D(mol.GetConformer().GetAtomPosition(int(nearest_atom_idx)).x, mol.GetConformer().GetAtomPosition(int(nearest_atom_idx)).y))
            ax.plot([start_coord.x, end_coord.x], [start_coord.y, end_coord.y], color="green", linewidth=10)

    # Colorbar
    norm = Normalize(vmin=0, vmax=1)  # Adjusted normalization
    cmap = plt.cm.Reds
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation="vertical")
    cbar.set_label("Attention")
    
    #Add title to my figure
    ax.set_title(prop_name+"___"+smi,fontsize=20)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    os.makedirs(save_dir+"combined/",exist_ok=True)
    plt.savefig(save_dir+"combined/"+smi.replace("/","|")+"_"+prop_name+"_"+str(idx))
    if with_hydrogen:
        save_dir += "with_H/"
    else:
        save_dir += "no_H/"
    save_dir= os.path.join(save_dir,prop_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir+="/"+smi.replace("/","|")+"_"+prop_name+"_"+str(idx)
    try:
        plt.savefig(save_dir)  # Save with a transparent background
        plt.close()
    except:
        pass