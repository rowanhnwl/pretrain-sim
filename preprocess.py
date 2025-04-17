from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from itertools import product
from tqdm import tqdm
from math import ceil
import random
import os
from multiprocess import Pool, Manager

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

def get_best_smiles_pairs(dataset):

    smiles = list(dataset.keys())
    smiles_pairs = get_all_smiles_pairs(smiles)

    # Only use some pairs for the sake of efficiency
    # sample_rate = 0.999
    # n_sample = int(sample_rate * len(smiles_pairs))
    # smiles_pairs = random.sample(smiles_pairs, n_sample)

    # Split the pairs into lists for subprocessing
    n_cpu = os.cpu_count()
    smiles_pairs_splits = split_to_sublists(smiles_pairs, n_cpu)
    
    with Pool(n_cpu) as p:
        shared_sim_list = manager.list()
        args_list = [(pairs_split, shared_sim_list, i) for i, pairs_split in enumerate(smiles_pairs_splits)]

        p.map(proc_sublist_pairs, args_list)

    sim_list = list(shared_sim_list)

    sim_thresh = 0.9
    smiles_pairs = []
    for x in sim_list:
        pdiff = dataset[x[0]] - dataset[x[1]]
        if x[2] >= sim_thresh and pdiff != 0:
            if pdiff > 0: # Greater prop val first
                smiles_pairs.append((x[0], x[1], pdiff))
            elif pdiff < 0:
                smiles_pairs.append((x[1], x[0], -pdiff))

    print(f"{len(smiles_pairs)} total pairs")

    return smiles_pairs

def get_all_smiles_pairs(smiles):
    pairs = []
    for i in range(len(smiles)):
        for k in range(i + 1, len(smiles)):
            pairs.append((smiles[i], smiles[k]))

    return pairs