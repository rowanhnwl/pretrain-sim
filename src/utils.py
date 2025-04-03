import pandas as pd
import os
import csv

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcTPSA

def calc_tpsa(smi):
    mol = Chem.MolFromSmiles(smi)

    if mol is not None:
        tpsa = CalcTPSA(mol)
    else:
        return None

    return tpsa

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

def split_property_dataset(data_dict, target, ref, tolerance):
    min_target_val = target - tolerance
    max_target_val = target + tolerance

    min_ref_val = ref - tolerance
    max_ref_val = ref + tolerance

    refs = []
    targets = []

    for smi, val in data_dict.items():
        if val >= min_target_val and val <= max_target_val:
            targets.append(smi)
        elif val >= min_ref_val and val <= max_ref_val:
            refs.append(smi)

    return refs, targets