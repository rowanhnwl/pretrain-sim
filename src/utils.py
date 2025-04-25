import pandas as pd
import os
import csv
import json

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem.Crippen import MolLogP

from datasets import Dataset

import torch

def calc_tpsa(smi):
    mol = Chem.MolFromSmiles(smi)

    if mol is not None:
        tpsa = CalcTPSA(mol)
    else:
        return None

    return tpsa

def calc_xlogp(smi):
    mol = Chem.MolFromSmiles(smi)

    if mol is not None:
        xlogp = MolLogP(mol)
    else:
        return None

    return xlogp

def load_zinc250k_all(path="data/zinc250k"):
    full_path = os.path.join(path, os.listdir(path)[0])
    
    all_props = ["logP", "qed", "SAS"]

    smiles = []
    with open(full_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            smiles.append((row["smiles"].strip(), float(row[all_props[0]]), float(row[all_props[1]]), float(row[all_props[2]])))

    return smiles

def load_zinc250k(path="data/zinc250k", prop=None):
    full_path = os.path.join(path, os.listdir(path)[0])
    if prop is None:
        df = pd.read_csv(full_path)

        smiles = [smi.strip() for smi in list(df[df.columns[0]])]
    elif prop != "tpsa":
        smiles = []
        with open(full_path, "r", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                smiles.append((row["smiles"].strip(), float(row[prop])))
    else:
        df = pd.read_csv(full_path)

        smiles_only = [smi.strip() for smi in list(df[df.columns[0]])]
        
        smiles = []
        for smi in smiles_only:
            tpsa = calc_tpsa(smi)
            if tpsa is not None:
                smiles.append((smi, tpsa))

    return smiles

def load_prop_json(path):
    with open(path, "r") as f:
        data_dict = json.load(f)

    return [(smi, pval) for (smi, pval) in data_dict.items()]

def split_property_dataset(data_dict, t_range, r_range):
    min_target_val, max_target_val = t_range
    min_ref_val, max_ref_val = r_range

    refs = []
    targets = []

    for smi, val in data_dict.items():
        if val >= min_target_val and val <= max_target_val:
            targets.append(smi)
        elif val >= min_ref_val and val <= max_ref_val:
            refs.append(smi)

    return refs, targets

def split_train_valid(dataset, valid_ratio):
    tokenized_dataset_list = dataset.data

    dataset = Dataset.from_list(tokenized_dataset_list)
    dataset_split = dataset.train_test_split(test_size=valid_ratio)

    train_dataset = dataset_split["train"]
    valid_dataset = dataset_split["test"]

    return train_dataset, valid_dataset