import pandas as pd
import os

def load_zinc250k(path="zinc250k"):
    full_path = os.path.join(path, os.listdir(path)[0])
    df = pd.read_csv(full_path)

    smiles = [smi.strip() for smi in list(df[df.columns[0]])]

    return smiles