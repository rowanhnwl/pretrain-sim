from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors

from src.prop_pretraining.chemfm import LlamaForPropPred

import torch
import os
import shutil
import csv
import subprocess
import pandas as pd

device = ("cuda" if torch.cuda.is_available() else "cpu")

prop_to_model_dict = {
    "caco2_permeability": "checkpoints/eval/caco2_permeability/model_0/best.pt",
    "acute_toxicity": "PATH",
    "lipophilicity": "PATH",
    "solubility": "PATH"
}

def write_csv_data(data, path):

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)

def infer_prop_vals(smiles, model_path):

    os.makedirs("tmp", exist_ok=True)

    smiles_csv_format = [[smi] for smi in smiles if Chem.MolFromSmiles(smi) is not None]
    csv_data = [["SMILES"]] + smiles_csv_format
    gen_path = "tmp/gen_smiles.csv"
    out_path = "tmp/preds.csv"

    write_csv_data(csv_data, gen_path)

    subprocess.run(
        f"  chemprop predict \
            --test-path \"{gen_path}\" \
            --model-path \"{model_path}\" \
            --preds-path \"{out_path}\"",
        shell=True
    )

    df = pd.read_csv(out_path)
    pred_vals = list(df[df.columns[1]])

    shutil.rmtree("tmp")

    smiles_metrics_dict = {smi[0]: val for (smi, val) in zip(smiles_csv_format, pred_vals)}

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

            smiles_metrics_dict[smi] = val

    return smiles_metrics_dict

def eval_prop(smiles, prop):
    if prop in prop_to_model_dict.keys():
        model_path = prop_to_model_dict[prop]

        prop_dict = infer_prop_vals(smiles, model_path)

    else:
        prop_dict = calc_prop_vals(smiles, prop)

    return prop_dict