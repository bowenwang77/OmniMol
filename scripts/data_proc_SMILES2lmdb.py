import pandas as pd
import lmdb
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import io
import glob
import os
from concurrent.futures import ProcessPoolExecutor
import pickle
from multiprocessing import Pool
from tqdm import tqdm

def get_property_name(list_of_names):
    for name in list_of_names:
        if name != 'smiles' and name != 'group':
            return name

def smi2_2Dcoords(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(coordinates), "2D coordinates shape is not align with {}".format(smi)
    return coordinates


def smi2_3Dcoords(smi,cnt):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    coordinate_list=[]
    for seed in range(cnt):
        try:
            res = AllChem.EmbedMolecule(mol, randomSeed=seed)  # will random generate conformer with seed equal to -1. else fixed random seed.
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)       # some conformer can not use MMFF optimize
                    coordinates = mol.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi)            
                    
            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(smi)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_tmp)       # some conformer can not use MMFF optimize
                    coordinates = mol_tmp.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi) 
        except:
            print("Failed to generate 3D, replace with 2D")
            coordinates = smi2_2Dcoords(smi) 

        assert len(mol.GetAtoms()) == len(coordinates), "3D coordinates shape is not align with {}".format(smi)
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list



def inner_smi2coords(content):
    smi = content[0]
    target = content[1]
    property_name = content[2]
    local_idx = content[3]
    cnt = 10 # conformer num,all==11, 10 3d + 1 2d

    mol = Chem.MolFromSmiles(smi)
    if len(mol.GetAtoms()) > 400:
        coordinate_list =  [smi2_2Dcoords(smi)] * (cnt+1)
        print("atom num >400,use 2D coords",smi)
    else:
        coordinate_list = smi2_3Dcoords(smi,cnt)
        coordinate_list.append(smi2_2Dcoords(smi).astype(np.float32))
    mol = AllChem.AddHs(mol)
    atoms = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])  # after add H 
    print("writing:  ", property_name ,local_idx, smi, target)
    return pickle.dumps({'atoms': atoms, 
    'coordinates': coordinate_list, 
    'mol':mol,'smi': smi, 'target': target, 'local_idx': local_idx, 'property_name': property_name}, protocol=-1)

def smi2coords(content):
    try:
        return inner_smi2coords(content)
    except:
        print("failed smiles: {}".format(content[0]))
        return None
    
def write_lmdb(inpath='./', outpath='./', nthreads=16):

    df = pd.read_csv(os.path.join(inpath))
    sz = len(df)
    property_name = get_property_name(list(df.keys()))
    df['property_name']=property_name
    df['local_idx']=df.index
    train, valid, test = df[df['group']=="training"], df[df['group']=="val"], df[df['group']=="test"]

    for name, content_list in [('train', zip(*[train[c].values.tolist() for c in ['smiles',property_name, 'property_name', 'local_idx'] ])),
                                ('val_id', zip(*[valid[c].values.tolist() for c in ['smiles',property_name, 'property_name', 'local_idx']])),
                                ('test_id', zip(*[test[c].values.tolist() for c in ['smiles',property_name, 'property_name', 'local_idx']]))]:
        os.makedirs(os.path.join(outpath, name), exist_ok=True)
        output_name = os.path.join(outpath, name,"data.lmdb")
        try:
            os.remove(output_name)
        except:
            pass
        env_new = lmdb.open(
            output_name,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(100e9),
        )
        txn_write = env_new.begin(write=True)
        with Pool(nthreads) as pool:
            i = 0
            for inner_output in tqdm(pool.imap(smi2coords, content_list)):
                if inner_output is not None:
                    txn_write.put(f'{i}'.encode("ascii"), inner_output)
                    i += 1
            print('{} process {} lines'.format(name, i))
            txn_write.commit()
            env_new.close()
        print('finish writing {}'.format(output_name))

def process_all_csv_files(input_folder, output_prefix, output_root_dir):
    for file_path in glob.glob(os.path.join(input_folder, '*.csv')):
        write_lmdb(inpath=file_path, outpath=output_root_dir+output_prefix+file_path.split("/")[-1].split(".")[0], nthreads=32)

def combine_lmdb_files(lmdb_dirs, lmdb_files, output_dir, combined_lmdb_files):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for lmdb_file, combined_lmdb_file in zip(lmdb_files, combined_lmdb_files):
        if not os.path.exists(os.path.join(output_dir, combined_lmdb_file)):
            # os.mkdir(output_dir)
            os.makedirs(os.path.join(output_dir, combined_lmdb_file), exist_ok=True)
        combined_env = lmdb.open(os.path.join(output_dir, combined_lmdb_file, "data.lmdb"),subdir=False, map_size=int(1e12))
        combined_txn = combined_env.begin(write=True)
        idx = 0
        print("Combining {}".format(lmdb_file))
        for lmdb_dir in lmdb_dirs:
            if lmdb_dir.split("/")[-1] == "lmdb_combined":
                continue
            lmdb_path = os.path.join(lmdb_dir, lmdb_file)
            env = lmdb.open(lmdb_path,subdir=False, readonly=True)
            txn = env.begin()
            print("Processing {}".format(lmdb_path))
            with txn.cursor() as cursor:
                for key, value in cursor:
                    instance = pickle.loads(value)
                    instance['idx'] = idx  # Add the 'idx' property to the instance
                    updated_value = pickle.dumps(instance)
                    
                    combined_key = f"{idx}".encode("ascii")
                    combined_txn.put(combined_key, updated_value)
                    idx += 1

            env.close()

        combined_txn.commit()
        combined_env.close()

output_prefix = ""
##Single proces
input_folder = "example"

# uff = AllChem.UFFGetMoleculeForceField
# etkdg = True
num_conformers = 10
root_dir = input_folder + "/lmdb/"
process_all_csv_files(input_folder, output_prefix, root_dir)

##Process folder
# input_folder_root = "/DEFAULT_NEW/Code/DRFormer_ADMET/graphormer/admet_paper/1226_node_attention/bowen_sets_1227"
# subdirectories = [os.path.join(input_folder_root, d) for d in os.listdir(input_folder_root) if os.path.isdir(os.path.join(input_folder_root, d))]
# subdirectories = [
#     "/DEFAULT_NEW/Code/DRFormer_ADMET/graphormer/admet_paper/1226_node_attention/bowen_sets_1227/Ames_Carcinogenicity_CYP3A4-inh_CYP1A2-inh_45",
#     "/DEFAULT_NEW/Code/DRFormer_ADMET/graphormer/admet_paper/1226_node_attention/bowen_sets_1227/Ames_Carcinogenicity_F(20%)_F(30%)_31",
#     "/DEFAULT_NEW/Code/DRFormer_ADMET/graphormer/admet_paper/1226_node_attention/bowen_sets_1227/Ames_Carcinogenicity_H-HT_DILI_20",
#     "/DEFAULT_NEW/Code/DRFormer_ADMET/graphormer/admet_paper/1226_node_attention/bowen_sets_1227/Ames_Carcinogenicity_Pgp-inh_Pgp-sub_5",
# ]
# for subdir in subdirectories:
#     root_dir = subdir + "/lmdb/"
#     process_all_csv_files(subdir, output_prefix, root_dir)

