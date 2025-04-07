from tqdm import tqdm
import csv

from torch.utils.data import Dataset as TDataset
from datasets import Dataset as DDataset

EOS_TOKEN_ID = 265

class ZincSMILESDataset(TDataset):
    def __init__(self, n, tokenizer, path="data/zinc250k/250k_rndm_zinc_drugs_clean_3.csv", max_len=512):
        self.filepath = path
        self.n = n
        self.max_len = max_len
        self.tokenizer=tokenizer

        self.data = []
        self.load_from_csv()

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

                    if len(smi) < self.max_len:
                        tokenized_smi = self.tokenizer(smi)
                        del tokenized_smi["token_type_ids"]

                        tokenized_smi["input_ids"].append(EOS_TOKEN_ID)
                        tokenized_smi["attention_mask"].append(1)

                        self.data.append(tokenized_smi)
                        pbar.update(1)

                    if len(self.data) == self.n:
                        break

def split_train_valid(dataset, valid_ratio):
    tokenized_dataset_list = dataset.data

    dataset = DDataset.from_list(tokenized_dataset_list)
    dataset_split = dataset.train_test_split(test_size=valid_ratio)

    train_dataset = dataset_split["train"]
    valid_dataset = dataset_split["test"]

    return train_dataset, valid_dataset