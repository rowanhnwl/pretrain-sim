from src.generation.latent import *
from src.generation.causal import *
from src.utils import *

from src.prop_pretraining.chemfm import LlamaForPropPred

from preprocess import get_best_smiles_pairs

from tqdm import tqdm
import torch
import json

import seaborn as sns

device = ("cuda" if torch.cuda.is_available() else "cpu")

def main():
    dataset_path = "data/props/xlogp.json"
    
    s_i = -0.001
    s_f = -30.0

    with open(dataset_path, "r") as f:
        data_dict = json.load(f)

    model, tokenizer = load_causal_lm_and_tokenizer(
        model_path="output/final"
    )

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    #smiles_pairs = get_best_smiles_pairs(dataset)
    with open("smiles_pairs.json", "r") as f:
        smiles_pairs = json.load(f)["pairs"]

    icv = pair_based_icv(smiles_pairs, model, tokenizer)

    config_steering(icv, s_i, s_f, model)
    
    smiles = []
    for _ in tqdm(range(20)):
        smiles += generate_smiles(
            "C",
            5,
            model,
            tokenizer,
        )   
    
    tpsas = []
    s = []
    for smi in smiles:
        tpsa = calc_xlogp(smi)

        if tpsa and tpsa > min(data_dict.values()) and tpsa < max(data_dict.values()):
            tpsas.append(tpsa)
            s.append(smi)
    sns.kdeplot(tpsas, bw_adjust=0.5)
    plt.savefig("dist.png")

    print(sum(tpsas)/len(tpsas))

    #print(list(zip(s, tpsas)))
if __name__ == "__main__":
    main()