from tqdm import tqdm
import lmdb
import pickle

from torch.utils.data import Dataset as TDataset
from datasets import Dataset as DDataset

EOS_TOKEN_ID = 265

class SMILESDataset(TDataset):
    def __init__(self, filepath, n):
        self.filepath = filepath
        self.data = []

        self.init_from_lmdb(filepath, n)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def init_from_lmdb(self, filepath, n):
        env = lmdb.open(filepath, map_size=1024 ** 4, lock=False, subdir=False)

        with env.begin() as txn:
            for i in tqdm(range(n), desc="Loading tokenized SMILES"):
                key_str = str(i).encode("utf-8")

                encoded_ts = txn.get(key_str)

                if encoded_ts is None:
                    break

                tokenized_smiles = pickle.loads(encoded_ts)
                self.data.append(tokenized_smiles)

        env.close()

def split_train_valid(dataset, valid_ratio):
    tokenized_dataset_list = dataset.data

    dataset = DDataset.from_list(tokenized_dataset_list)
    dataset_split = dataset.train_test_split(test_size=valid_ratio)

    train_dataset = dataset_split["train"]
    valid_dataset = dataset_split["test"]

    return train_dataset, valid_dataset