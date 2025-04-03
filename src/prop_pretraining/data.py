import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import random

PAD_TOKEN_ID = 266

class PropDataset(Dataset):
    def __init__(self, master_data_list, n_samples, tokenizer, n_props=1, max_len=512):
        self.master_data_list = master_data_list
        self.n_samples = n_samples
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.n_props = n_props

        self.smiles_prop_data = self.sample_and_tokenize()
        self.standardize()

    def __len__(self):
        return len(self.smiles_prop_data)
    
    def __getitem__(self, index):
        return self.smiles_prop_data[index]

    def sample_and_tokenize(self):
        sampled_smiles = random.sample(self.master_data_list, self.n_samples)

        smiles_prop_data = []
        for sample in sampled_smiles:
            smi = sample[0]
            props = list(sample[1:]) if self.n_props > 1 else sample[1]

            tokenized_smi = self.tokenize_smiles(smi)

            if len(smi) < self.max_len:
                smiles_prop_data.append({
                    "smi": tokenized_smi,
                    "prop": props
                })

        return smiles_prop_data

    def tokenize_smiles(self, smi):
        tokenized = self.tokenizer(smi)
        del tokenized["token_type_ids"]

        return tokenized
    
    def standardize(self):
        prop_vals = []
        for x in self.smiles_prop_data:
            if self.n_props > 1:
                prop_vals.append(x["prop"])
            else:
                prop_vals.append([x["prop"]])

        prop_vals = torch.tensor(prop_vals)
        
        mu = prop_vals.mean(dim=0, keepdim=True)
        sigma = prop_vals.std(dim=0, keepdim=True)

        for x in self.smiles_prop_data:
            x["prop"] = ((torch.tensor(x["prop"]) - mu) / sigma).tolist()
    
def collate_batch(batch):
    smi_token_ids = []
    smi_attn_masks = []

    props = []

    for smi_with_prop in batch:
        smi_data = smi_with_prop["smi"]
        prop = smi_with_prop["prop"]

        smi_token_ids.append(torch.tensor(smi_data["input_ids"]).squeeze(0))
        smi_attn_masks.append(torch.tensor(smi_data["attention_mask"]).squeeze(0))

        props.append(prop)

    smi_token_ids = pad_sequence(smi_token_ids, batch_first=True, padding_value=PAD_TOKEN_ID).to(torch.long)
    smi_attn_masks = pad_sequence(smi_attn_masks, batch_first=True, padding_value=0).to(torch.long)

    return {
        "smi_input_ids": smi_token_ids,
        "smi_attn_mask": smi_attn_masks,
        "prop": torch.tensor(props, dtype=torch.float)
    }