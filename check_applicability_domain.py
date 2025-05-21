import os
import sys
import argparse
import pandas as pd
import pickle
import numpy as np
import time
from rdkit import Chem
import multiprocessing
from functools import partial

def calculate_descriptors(mol):
    """Calculate molecular descriptors, without using cache"""
    if mol is None:
        return None
    
    try:
        from rdkit.Chem import Descriptors, AllChem, Lipinski
        from rdkit.Chem import GraphDescriptors as GD
        from rdkit.Chem.EState import EState_VSA
        from rdkit.Chem.Crippen import MolLogP
        from rdkit.Chem.rdMolDescriptors import CalcNumRings, CalcNumRotatableBonds, CalcTPSA
        
        # Basic physicochemical properties
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        rb = Descriptors.NumRotatableBonds(mol)
        rc = Descriptors.RingCount(mol)
        
        # Topological descriptors
        bertz = GD.BertzCT(mol)
        chi1 = Descriptors.Chi1(mol)
        chi1n = Descriptors.Chi1n(mol)
        
        # Electronic descriptors
        estate_sum = sum(EState_VSA.EState_VSA_(mol))
        peoe_vsa1 = Descriptors.PEOE_VSA1(mol)
        peoe_vsa2 = Descriptors.PEOE_VSA2(mol)
        
        # Stereocenter and symmetry
        chiral_centers = len(Chem.FindMolChiralCenters(mol))
        
        # Complexity descriptors
        fsp3 = Descriptors.FractionCSP3(mol)
        qed = Descriptors.qed(mol)
        
        # Aromaticity
        aromatic_rings = Lipinski.NumAromaticRings(mol)
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        
        # Functional group descriptors
        num_heterocycles = Lipinski.NumHeteroatoms(mol)
        
        # Atom properties
        heavy_atom_count = Descriptors.HeavyAtomCount(mol)
        
        features = [
            mw, logp, hbd, hba, tpsa, rb, rc,
            bertz, chi1, chi1n, 
            estate_sum, peoe_vsa1, peoe_vsa2,
            chiral_centers, fsp3, qed,
            aromatic_rings, aromatic_atoms,
            num_heterocycles, heavy_atom_count
        ]
        
        return features
    except Exception as e:
        print(f"Error calculating descriptors: {e}")
        return None

def is_in_domain(smiles, model, threshold):
    """Determine if a molecule is within the applicability domain"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, None
    
    features = calculate_descriptors(mol)
    if features is None:
        return False, None
    
    features = np.array([features])
    
    anomaly_score = model.decision_function(features)[0]
    is_in_ad = anomaly_score >= threshold
    
    return is_in_ad, anomaly_score

def process_domain_check(data):
    """Process applicability domain check for a single molecule"""
    smiles, model, threshold = data
    try:
        is_in_ad, score = is_in_domain(smiles, model, threshold)
        return (smiles, is_in_ad, score)
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return (smiles, False, None)

def is_in_domain_batch(smiles_list, model, threshold, n_jobs=None):
    """Parallel check if multiple molecules are within the applicability domain"""
    
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    
    # Create data packages containing all required parameters
    data_list = [(smiles, model, threshold) for smiles in smiles_list]
    
    # Create process pool
    pool = multiprocessing.Pool(processes=n_jobs)
    
    # Use process pool for parallel processing
    results = pool.map(process_domain_check, data_list)
    
    # Close process pool
    pool.close()
    pool.join()
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Check if SMILES are within the applicability domain')
    parser.add_argument('--input', required=True, help='Path to CSV file containing SMILES')
    parser.add_argument('--endpoint', required=True, help='Endpoint name, used to load the corresponding applicability domain model')
    parser.add_argument('--smiles-col', default='smiles', help='Name of the SMILES column in the CSV file, default is "smiles"')
    parser.add_argument('--output', help='Path to output CSV file, default is input filename_ad_result.csv')
    parser.add_argument('--model-path', default='./iso_forest/ad_models', help='Path where applicability domain models are saved')
    parser.add_argument('--n-jobs', type=int, default=None, help='Number of CPU cores to use for parallel processing, default is CPU count-1')
    parser.add_argument('--all-model', action='store_true', help='Also use the ALL model for checking')
    
    args = parser.parse_args()
    
    # Set output file path
    if not args.output:
        input_base = os.path.basename(args.input)
        input_name = os.path.splitext(input_base)[0]
        args.output = f"{input_name}_{args.endpoint}_ad_result.csv"
    
    # Load model
    model_file = os.path.join(args.model_path, f"{args.endpoint}_ad_model.pkl")
    if not os.path.exists(model_file):
        print(f"Error: Model file not found {model_file}")
        return
    
    print(f"Loading {args.endpoint} model...")
    with open(model_file, 'rb') as f:
        model_info = pickle.load(f)
        model = model_info['model']
        threshold = model_info['threshold']
    
    # If ALL model is specified, load it too
    all_model = None
    all_threshold = None
    if args.all_model:
        all_model_file = os.path.join(args.model_path, "ALL_ad_model.pkl")
        if os.path.exists(all_model_file):
            print("Loading ALL model...")
            with open(all_model_file, 'rb') as f:
                all_model_info = pickle.load(f)
                all_model = all_model_info['model']
                all_threshold = all_model_info['threshold']
        else:
            print("Warning: ALL model not found, will only use the specified endpoint model")
    
    # Read SMILES data
    try:
        df = pd.read_csv(args.input)
        if args.smiles_col not in df.columns:
            print(f"Error: Input file does not contain column named '{args.smiles_col}'")
            return
    except Exception as e:
        print(f"Error: Cannot read input file: {e}")
        return
    
    # Get SMILES list
    smiles_list = df[args.smiles_col].tolist()
    total_smiles = len(smiles_list)
    print(f"Read {total_smiles} SMILES")
    
    # Check if in applicability domain
    start_time = time.time()
    print(f"Checking applicability domain using {args.endpoint} model...")
    results = is_in_domain_batch(
        smiles_list, 
        model, 
        threshold, 
        n_jobs=args.n_jobs
    )
    
    # Extract results
    in_domain = [result[1] for result in results]
    anomaly_scores = [result[2] for result in results]
    
    # Add to dataframe
    df['in_domain'] = in_domain
    df['anomaly_score'] = anomaly_scores
    
    # If ALL model is specified, also use it for checking
    if all_model is not None:
        print("Checking applicability domain using ALL model...")
        all_results = is_in_domain_batch(
            smiles_list, 
            all_model, 
            all_threshold, 
            n_jobs=args.n_jobs
        )
        
        # Extract results
        all_in_domain = [result[1] for result in all_results]
        all_anomaly_scores = [result[2] for result in all_results]
        
        # Add to dataframe
        df['in_domain_all'] = all_in_domain
        df['anomaly_score_all'] = all_anomaly_scores
    
    # Add calculation result statistics
    in_domain_count = sum(in_domain)
    out_domain_count = total_smiles - in_domain_count
    in_domain_ratio = in_domain_count / total_smiles if total_smiles > 0 else 0
    
    print(f"\nResult statistics:")
    print(f"Total molecules: {total_smiles}")
    print(f"In Domain: {in_domain_count} ({in_domain_ratio:.2%})")
    print(f"Out of Domain: {out_domain_count} ({1-in_domain_ratio:.2%})")
    
    if all_model is not None:
        all_in_domain_count = sum(all_in_domain)
        all_out_domain_count = total_smiles - all_in_domain_count
        all_in_domain_ratio = all_in_domain_count / total_smiles if total_smiles > 0 else 0
        
        print("\nALL model result statistics:")
        print(f"In Domain: {all_in_domain_count} ({all_in_domain_ratio:.2%})")
        print(f"Out of Domain: {all_out_domain_count} ({1-all_in_domain_ratio:.2%})")
    
    # Save results
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")
    
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
