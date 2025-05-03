from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, Descriptors, QED
from rdkit.Chem import AllChem, DataStructs

from src.prop_pretraining.chemfm import LlamaForPropPred
from src.generation.latent import get_all_hidden_states

import torch
import os
from math import ceil
from multiprocessing import Pool, Manager
import pandas as pd
import json
import pickle
from tqdm import tqdm

from heapq import nlargest, nsmallest

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

manager = Manager()
device = ("cuda" if torch.cuda.is_available() else "cpu")
RDLogger.DisableLog('rdApp.*')

prop_to_dataset_dict = {
    "caco2_permeability": "data/props/caco2_permeability.json",
    "acute_toxicity": "data/props/acute_toxicity.json",
    "lipophilicity": "data/props/lipophilicity.json",
    "solubility": "data/props/solubility.json"
}

global_data_dict = None
global_states_dict = None
global_gen_states_dict = None

fpgen = AllChem.GetRDKitFPGenerator()

def init_worker(states, gen_states, data_dict):
    global global_data_dict
    global global_states_dict
    global global_gen_states_dict
    global_data_dict = data_dict
    global_states_dict = states
    global_gen_states_dict = gen_states

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

    smiles = args

    metrics_dict = {}

    dataset_smiles = list(global_data_dict.keys())

    for smi in tqdm(smiles):
        sims = []
        vals = []
        for d_smi in dataset_smiles:
            sim_score = tanimoto_similarity(d_smi, smi)

            if sim_score is not None:
                sims.append(sim_score)
                vals.append(global_data_dict[d_smi])

        if len(sims) > 0:
            sim_val_pairs = nlargest(5, list(zip(sims, vals)), key=lambda x: x[0])
            print(sim_val_pairs)
            close_vals = [x[1] for x in sim_val_pairs]
            est_val = sum(close_vals) / len(close_vals)

            metrics_dict[smi] = est_val

    return metrics_dict

def state_infer_proc(args):

    smiles = args

    metrics_dict = {}

    dataset_smiles = list(global_data_dict.keys())

    for smi in tqdm(smiles):
        sims = []
        vals = []
        for d_smi in dataset_smiles:
            sim_score = torch.cdist(global_states_dict[d_smi], global_gen_states_dict[smi]).item()

            if sim_score is not None:
                sims.append(sim_score)
                vals.append(global_data_dict[d_smi])

        if len(sims) > 0:
            sim_val_pairs = nsmallest(5, list(zip(sims, vals)), key=lambda x: x[0])
            print(sim_val_pairs)
            close_vals = [x[1] for x in sim_val_pairs]
            est_val = sum(close_vals) / len(close_vals)

            metrics_dict[smi] = est_val

    return metrics_dict

def structure_inference(smiles, dataset_path, gen_states_dict, states_path):

    with open(states_path, "rb") as f:
        states_dict = pickle.load(f)

    last_layer = list(states_dict.keys())[-1]

    last_states_dict = {}
    for (smi, state) in states_dict[last_layer].items():
        last_states_dict[smi] = state
    del states_dict

    with open(dataset_path, "r") as f:
        data_dict = json.load(f)
    
    n_cpu = os.cpu_count()
    smiles_splits = split_to_sublists(smiles, n_cpu)

    with Pool(n_cpu, initializer=init_worker, initargs=(last_states_dict, gen_states_dict, data_dict)) as p:
        args_list = [smiles_split for smiles_split in smiles_splits]

        metrics_dicts = p.map(structure_inference_proc, args_list)

    smiles_metrics_dict = {}
    for split_metrics_dict in metrics_dicts:
        smiles_metrics_dict.update(split_metrics_dict)

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
            elif prop == "qed":
                val = QED.qed(mol)

            smiles_metrics_dict[smi] = val

    return smiles_metrics_dict

def eval_prop(smiles, prop, gen_states_dict, states_path):
    if prop in prop_to_dataset_dict.keys():
        dataset_path = prop_to_dataset_dict[prop]

        prop_dict = structure_inference(smiles, dataset_path, gen_states_dict, states_path)

    else:
        prop_dict = calc_prop_vals(smiles, prop)

    return prop_dict