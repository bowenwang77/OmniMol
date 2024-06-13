import pandas as pd
import os
from rdkit import Chem

def read_csv_with_index(file_path):
    df = pd.read_csv(file_path)
    df.reset_index(inplace=True)
    return df

def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, canonical=True)
    return None

def find_common_smiles_and_output_csv(input_folder, subtask_names, output_folder):
    # csv_folder_path = '/HOME/DRFormer_ADMET/graphormer/admet_paper/intersect_SMILES/Group2'
    file_paths = [f"{input_folder}/{subtask}.csv" for subtask in subtask_names]

    first_df = read_csv_with_index(file_paths[0])
    first_df['canonical_smiles'] = first_df['smiles'].apply(canonicalize_smiles)
    common_smiles = set(first_df['canonical_smiles'])

    for file_path in file_paths[1:]:
        df = read_csv_with_index(file_path)
        df['canonical_smiles'] = df['smiles'].apply(canonicalize_smiles)
        current_smiles = set(df['canonical_smiles'])
        common_smiles = common_smiles.intersection(current_smiles)
    output_folder = output_folder + "_".join(subtask_names) +"_"+str(len(common_smiles))
    os.makedirs(output_folder,exist_ok = True)
    for file_path in file_paths:
        df = read_csv_with_index(file_path)
        df['smiles'] = df['smiles'].apply(canonicalize_smiles)
        screened_df = df[df['smiles'].isin(common_smiles)][['smiles', 'group', file_path.split("/")[-1][:-4]]]
        screened_df['group'] = 'training'
        val_df = screened_df.copy()
        val_df['group'] = 'val'
        test_df = screened_df.copy()
        test_df['group'] = 'test'

        final_df = pd.concat([screened_df, val_df, test_df])

        output_file = os.path.join(output_folder, f'screened_{os.path.splitext(os.path.basename(file_path))[0]}.csv')
        final_df.to_csv(output_file, index=False)
        print("Saving", output_file,"with molecule number:", len(screened_df))

# Example usage:
input_folder = "data/example"
# subtask_names = ['H-HT', 'hERG', 'EI', 'LogP']
subtask_names = "prop1/prop2".split("/")
#"Pgp-inh/Pgp-sub/H-HT/DILI/EC/EI/CYP3A4-inh/CYP1A2-inh/Ames/Carcinogenicity"
output_folder = 'intersect_of_mol/'
find_common_smiles_and_output_csv(input_folder, subtask_names, output_folder)
