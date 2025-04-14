from tqdm import tqdm
import csv

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcTPSA

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

EOS_TOKEN_ID = 265

class ZincSMILESWithTPSA(Dataset):
    def __init__(self, n, tokenizer, path="data/zinc250k/250k_rndm_zinc_drugs_clean_3.csv", max_len=512):
        self.filepath = path
        self.n = n
        self.max_len = max_len
        self.tokenizer=tokenizer

        self.data = []
        self.load_from_csv()
        self.standardize()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_from_csv(self):

        with tqdm(total=self.n, desc="Tokenizing SMILES strings") as pbar:
            with open(self.filepath, "r", newline="") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    smi = row["smiles"].strip()
                    tpsa = self.calc_tpsa(smi)

                    if len(smi) < self.max_len and tpsa:
                        tokenized_smi = self.tokenizer(smi)
                        del tokenized_smi["token_type_ids"]

                        tokenized_smi["input_ids"].append(EOS_TOKEN_ID)
                        tokenized_smi["attention_mask"].append(1)

                        self.data.append({
                            "input_ids": tokenized_smi["input_ids"],
                            "attn_mask": tokenized_smi["attention_mask"],
                            "labels": tokenized_smi["input_ids"],
                            "tpsa": tpsa
                        })
                        pbar.update(1)

                    if len(self.data) == self.n:
                        break

    def standardize(self):
        tpsas = torch.tensor([x["tpsa"] for x in self.data])
        
        std = tpsas.std()
        mean = tpsas.mean()

        for x in self.data:
            x["tpsa"] = (x["tpsa"] - mean) / std

    def calc_tpsa(self, smi):
        mol = Chem.MolFromSmiles(smi)

        if mol is not None:
            tpsa = CalcTPSA(mol)
        else:
            return None

        return tpsa

class MultiTaskCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        attention_mask = [torch.tensor(item["attn_mask"]) for item in batch]
        tpsa_values = torch.tensor([item["tpsa"] for item in batch], dtype=torch.float)

        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        labels = padded_input_ids.clone()

        return {
            "input_ids": padded_input_ids,
            "attn_mask": padded_attention_mask,
            "labels": labels,
            "tpsa": tpsa_values
        }