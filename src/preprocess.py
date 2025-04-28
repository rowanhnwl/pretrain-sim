from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from itertools import product
from tqdm import tqdm
from math import ceil
import random
from copy import deepcopy
import os
from multiprocess import Pool, Manager
from collections import Counter
from heapq import nsmallest

manager = Manager()

def tanimoto_similarity(smi1, smi2):

    try:
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
    except:
        return 0.0

    # Get fingerprints
    fpgen = AllChem.GetRDKitFPGenerator()
    fp1 = fpgen.GetFingerprint(mol1)
    fp2 = fpgen.GetFingerprint(mol2)

    tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)

    return tanimoto

def proc_sublist_pairs(args):
    ref_target_pairs, shared_sim_list, split_ind = args

    for ref_smi, target_smi in tqdm(ref_target_pairs, total=len(ref_target_pairs), desc=f"Processing split {split_ind + 1}"):
        sim = tanimoto_similarity(ref_smi, target_smi)
        shared_sim_list.append((
            ref_smi,
            target_smi,
            sim
        ))

def split_to_sublists(master_list, n_splits):
    pairs_per_split = ceil(float(len(master_list)) / n_splits)
    splits_list = []

    for n in range(n_splits):

        start_ind = n * pairs_per_split
        end_ind = (n + 1) * pairs_per_split if n != n_splits - 1 else len(master_list)

        splits_list.append(master_list[start_ind:end_ind])

    assert sum([len(sl) for sl in splits_list]) == len(master_list), "Split for multiprocessing didn't work"

    return splits_list

def get_best_smiles_pairs(dataset, sample_rate, sim_thresh):

    smiles = list(dataset.keys())
    smiles_pairs = get_all_smiles_pairs(smiles)

    # Only use some pairs for the sake of efficiency
    if sample_rate != 1.0:
        n_sample = int(sample_rate * len(smiles_pairs))
        smiles_pairs = random.sample(smiles_pairs, n_sample)

    # Split the pairs into lists for subprocessing
    n_cpu = os.cpu_count()
    smiles_pairs_splits = split_to_sublists(smiles_pairs, n_cpu)
    
    with Pool(n_cpu) as p:
        shared_sim_list = manager.list()
        args_list = [(pairs_split, shared_sim_list, i) for i, pairs_split in enumerate(smiles_pairs_splits)]

        p.map(proc_sublist_pairs, args_list)

    sim_list = list(shared_sim_list)

    smiles_pairs = []
    for x in sim_list:
        pdiff = dataset[x[0]] - dataset[x[1]]
        if x[2] >= sim_thresh and pdiff != 0:
            if pdiff > 0: # Greater prop val first
                smiles_pairs.append((x[0], x[1], x[2], pdiff))
            elif pdiff < 0:
                smiles_pairs.append((x[1], x[0], x[2], -pdiff))

    smiles_pairs = remove_duplicate_smiles(smiles_pairs)

    print(f"{len(smiles_pairs)} total pairs")

    return smiles_pairs

def remove_duplicate_smiles(smiles_pairs):
    all_smiles = []
    for pair in smiles_pairs:
        all_smiles += [pair[0], pair[1]]

    count_dict = dict(Counter(all_smiles))

    new_smiles_pairs = []
    for smi, count in count_dict.items():
        if count > 1:
            best_sim = 0.0
            best_pair = None
            for pair in smiles_pairs:
                if (smi == pair[0] or smi == pair[1]) and pair[2] > best_sim and (count_dict[pair[0]] != 0 and count_dict[pair[1]] != 0):
                    if len(pair[0]) > 5 and len(pair[1]) > 5:
                        best_pair = deepcopy(pair)
                        best_sim = pair[2]

            if best_pair is not None:  
                new_smiles_pairs.append(best_pair)
                count_dict[best_pair[0]] = 0
                count_dict[best_pair[1]] = 0

    return new_smiles_pairs

def get_all_smiles_pairs(smiles):
    pairs = []
    for i in range(len(smiles)):
        for k in range(i + 1, len(smiles)):
            pairs.append((smiles[i], smiles[k]))

    return pairs

# Latent-based approach
import torch
import torch.nn.functional as F

device = ("cuda" if torch.cuda.is_available() else "cpu")

EOS_TOKEN_ID = 265

def cat_for_eos(tokenized):
    tokenized["input_ids"] = torch.cat(
        (
            tokenized["input_ids"],
            torch.tensor([[EOS_TOKEN_ID]], dtype=torch.long, device=device)
        ),
        dim=1
    )
    tokenized["attention_mask"] = torch.cat(
        (
            tokenized["attention_mask"],
            torch.tensor([[1]], dtype=torch.long, device=device)
        ),
        dim=1
    )

def get_all_hidden_states(smiles, model, tokenizer):
    smiles_tokenized = tokenizer(smiles, return_tensors="pt").to(device)
    del smiles_tokenized["token_type_ids"]
    cat_for_eos(smiles_tokenized)

    with torch.no_grad():
        fwd_pass = model.model(**smiles_tokenized, output_hidden_states=True)

    hidden_states = fwd_pass.hidden_states

    final_hidden_states_list = []

    for hs in hidden_states[1:]: # No embedding layer
        final_hidden_states_list.append(hs[:, -1, :].to("cpu"))

    fhs_tensor = torch.stack(final_hidden_states_list)

    return fhs_tensor

def get_last_hidden_state(smiles, model, tokenizer):
    smiles_tokenized = tokenizer(smiles, return_tensors="pt").to(device)
    del smiles_tokenized["token_type_ids"]
    cat_for_eos(smiles_tokenized)

    model.eval()
    with torch.no_grad():
        fwd_pass = model.model(**smiles_tokenized, output_hidden_states=True)

    hidden_states = fwd_pass.hidden_states

    lhs = hidden_states[-1][:, -1, :].to("cpu")

    return lhs

def proc_sublist_pairs_latent(args):
    ref_target_pairs, shared_sim_list, split_ind, lhs_dict = args

    for ref_smi, target_smi in tqdm(ref_target_pairs, total=len(ref_target_pairs), desc=f"Processing split {split_ind + 1}"):
        
        lhs1 = lhs_dict[ref_smi]
        lhs2 = lhs_dict[target_smi]
        
        sim = torch.cdist(lhs1, lhs2).item()
        shared_sim_list.append((
            ref_smi,
            target_smi,
            sim
        ))

def get_best_smiles_pairs_latent(dataset, sample_rate, n_pairs, model, tokenizer):
    smiles = list(dataset.keys())
    smiles_pairs = get_all_smiles_pairs(smiles)

    # Get the dictionary of latent reps
    lhs_dict = {smi: get_last_hidden_state(smi, model, tokenizer) for smi in tqdm(smiles)}

    # Only use some pairs for the sake of efficiency
    if sample_rate != 1.0:
        n_sample = int(sample_rate * len(smiles_pairs))
        smiles_pairs = random.sample(smiles_pairs, n_sample)

    # Split the pairs into lists for subprocessing
    n_cpu = os.cpu_count()
    smiles_pairs_splits = split_to_sublists(smiles_pairs, n_cpu)
    
    with Pool(n_cpu) as p:
        shared_sim_list = manager.list()
        args_list = [(pairs_split, shared_sim_list, i, lhs_dict) for i, pairs_split in enumerate(smiles_pairs_splits)]

        p.map(proc_sublist_pairs_latent, args_list)

    sim_list = list(shared_sim_list)

    n_closest_pairs = nsmallest(n_pairs, sim_list, key=lambda x: x[2])
    smiles_pairs = []
    for p in n_closest_pairs:
        pdiff = dataset[p[0]] - dataset[p[1]]
        if pdiff > 0: # Greater prop val first
            smiles_pairs.append((p[0], p[1], p[2], pdiff))
        elif pdiff < 0:
            smiles_pairs.append((p[1], p[0], p[2], -pdiff))

    print(f"{len(smiles_pairs)} total pairs")

    return smiles_pairs