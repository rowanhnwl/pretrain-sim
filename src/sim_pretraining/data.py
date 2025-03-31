import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

import random
from tqdm import tqdm
import math
import numpy as np
from matplotlib import pyplot as plt

PAD_TOKEN_ID = 266

class TanimotoDataset(Dataset):
    def __init__(self, master_smiles_list, n_samples, tokenizer, max_len=512):
        self.master_smiles_list = master_smiles_list
        self.n_samples = n_samples
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.fpgen = AllChem.GetRDKitFPGenerator()

        self.pair_sim_data = self.build_pairs_list()
        self.zero_mean_sim_scores()

    def __len__(self):
        return len(self.pair_sim_data)
    
    def __getitem__(self, index):
        return self.pair_sim_data[index]

    def build_pairs_list(self):
        pairs_set = set()
        pairs_with_sim_list = []

        with tqdm(total=self.n_samples, desc="Getting SMILES pairs") as pbar:
            while len(pairs_set) < self.n_samples:
                old_len = len(pairs_set)

                (smi1, smi2) = random.sample(self.master_smiles_list, 2)
                sim_score = self.tanimoto_similarity(smi1, smi2)

                if sim_score is not None and len(smi1) <= self.max_len and len(smi2) <= self.max_len:
                    pairs_set.add((smi1, smi2))

                new_len = len(pairs_set)
                if new_len > old_len and sim_score is not None:

                    adjusted_sim_score = sim_score

                    pairs_with_sim_list.append({
                        "smi1": self.tokenize_smiles(smi1),
                        "smi2": self.tokenize_smiles(smi2),
                        "sim": adjusted_sim_score
                    })
                    pbar.update(1)

        return pairs_with_sim_list

    def tokenize_smiles(self, smi):
        tokenized = self.tokenizer(smi, return_tensors='pt')
        del tokenized["token_type_ids"]

        return tokenized

    def tanimoto_similarity(self, smi1, smi2):
        try:
            mol1 = Chem.MolFromSmiles(smi1)
            mol2 = Chem.MolFromSmiles(smi2)
        except:
            return None

        fp1 = self.fpgen.GetFingerprint(mol1)
        fp2 = self.fpgen.GetFingerprint(mol2)

        tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)

        return tanimoto

    def zero_mean_sim_scores(self):
        sim_mean, sim_std = self.get_sim_stats()

        for x in self.pair_sim_data:
            x["sim"] = 2*(x["sim"] - sim_mean)

    def adjust_sim_score_linear(self, sim):
        return (2 * sim) - 1

    def adjust_sim_score_cos(self, sim):
        return -math.cos(math.pi * sim)
    
    def get_sim_stats(self):
        sim_list = [x["sim"] for x in self.pair_sim_data]
        sim_mean = np.mean(sim_list)
        sim_std = np.std(sim_list)

        return sim_mean, sim_std
    
    def sim_histogram(self):
        sim_list = [x["sim"] for x in self.pair_sim_data]
        plt.hist(sim_list, bins=100)
        plt.savefig("hist.png")
    
def collate_batch(batch):
    smi1_token_ids = []
    smi1_attn_masks = []

    smi2_token_ids = []
    smi2_attn_masks = []

    sim_scores = []

    for pair_with_sim in batch:
        smi1_data = pair_with_sim["smi1"]
        smi2_data = pair_with_sim["smi2"]
        sim_score = pair_with_sim["sim"]

        smi1_token_ids.append(smi1_data["input_ids"].squeeze(0))
        smi1_attn_masks.append(smi1_data["attention_mask"].squeeze(0))
        smi2_token_ids.append(smi2_data["input_ids"].squeeze(0))
        smi2_attn_masks.append(smi2_data["attention_mask"].squeeze(0))

        sim_scores.append(sim_score)

    smi1_token_ids = pad_sequence(smi1_token_ids, batch_first=True, padding_value=PAD_TOKEN_ID).to(torch.long)
    smi1_attn_masks = pad_sequence(smi1_attn_masks, batch_first=True, padding_value=0).to(torch.long)
    smi2_token_ids = pad_sequence(smi2_token_ids, batch_first=True, padding_value=PAD_TOKEN_ID).to(torch.long)
    smi2_attn_masks = pad_sequence(smi2_attn_masks, batch_first=True, padding_value=0).to(torch.long)

    return {
        "smi1_input_ids": smi1_token_ids,
        "smi1_attn_mask": smi1_attn_masks,
        "smi2_input_ids": smi2_token_ids,
        "smi2_attn_mask": smi2_attn_masks,
        "sim": torch.tensor(sim_scores, dtype=torch.float)
    }