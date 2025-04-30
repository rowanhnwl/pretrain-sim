from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors
from rdkit.Chem import AllChem, DataStructs

from src.prop_pretraining.chemfm import LlamaForPropPred

import torch
import os
from math import ceil
from multiprocessing import Pool, Manager
import pandas as pd
import json

manager = Manager()
device = ("cuda" if torch.cuda.is_available() else "cpu")

prop_to_dataset_dict = {
    "caco2_permeability": "data/props/caco2_permeability.json",
    "acute_toxicity": "data/props/acute_toxicity.json",
    "lipophilicity": "data/props/lipophilicity.json",
    "solubility": "data/props/solubility.json"
}

fpgen = AllChem.GetRDKitFPGenerator()

def tanimoto_similarity(smi1, smi2):

    try:
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
    except:
        return None

    if mol1 is None or mol2 is None:
        return None

    # Get fingerprints
    fp1 = fpgen.GetFingerprint(mol1)
    fp2 = fpgen.GetFingerprint(mol2)

    tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)

    return tanimoto

def split_to_sublists(master_list, n_splits):
    pairs_per_split = ceil(float(len(master_list)) / n_splits)
    splits_list = []

    for n in range(n_splits):

        start_ind = n * pairs_per_split
        end_ind = (n + 1) * pairs_per_split if n != n_splits - 1 else len(master_list)

        splits_list.append(master_list[start_ind:end_ind])

    assert sum([len(sl) for sl in splits_list]) == len(master_list), "Split for multiprocessing didn't work"

    return splits_list

def structure_inference_proc(args):

    (smiles, data_dict, shared_results_dict) = args

    dataset_smiles = list(data_dict.keys())

    for smi in smiles:
        sims = []
        vals = []
        for d_smi in dataset_smiles:
            sim_score = tanimoto_similarity(d_smi, smi)

            if sim_score is not None:
                sims.append(sim_score)
                vals.append(data_dict[d_smi])

        sims_squared_sum = sum([s**2 for s in sims])
        sims_normed = [s**2 / sims_squared_sum for s in sims]

        est_val = sum([x * y for (x, y) in zip(vals, sims_normed)])

        shared_results_dict[smi] = est_val

def structure_inference(smiles, dataset_path):
    with open(dataset_path, "r") as f:
        data_dict = json.load(f)
    
    n_cpu = os.cpu_count()
    smiles_splits = split_to_sublists(smiles, n_cpu)

    with Pool(n_cpu) as p:
        shared_results_dict = manager.dict()
        args_list = [(smiles_split, data_dict, shared_results_dict) for smiles_split in smiles_splits]

        p.map(structure_inference_proc, args_list)

    smiles_metrics_dict = dict(shared_results_dict)

    return smiles_metrics_dict

def calc_prop_vals(smiles, prop):
    smiles_metrics_dict = {}

    # Calculate the molecular weight
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)

        if mol:
            if prop == "molecular_weight":
                val = Descriptors.MolWt(mol)
            elif prop == "tpsa":
                val = rdMolDescriptors.CalcTPSA(mol)
            elif prop == "xlogp":
                val, mr = rdMolDescriptors.CalcCrippenDescriptors(mol)
            elif prop == "rotatable_bond_count":
                val = rdMolDescriptors.CalcNumRotatableBonds(mol)

            smiles_metrics_dict[smi] = val

    return smiles_metrics_dict

def eval_prop(smiles, prop):
    if prop in prop_to_dataset_dict.keys():
        dataset_path = prop_to_dataset_dict[prop]

        prop_dict = structure_inference(smiles, dataset_path)

    else:
        prop_dict = calc_prop_vals(smiles, prop)

    return prop_dict