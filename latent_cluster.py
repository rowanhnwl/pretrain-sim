from src.generation.latent import *
from src.generation.causal import *
from src.utils import split_property_dataset, calc_tpsa

from src.prop_pretraining.chemfm import LlamaForPropPred

from preprocess import get_best_smiles_pairs

import torch
import json

import seaborn as sns

device = ("cuda" if torch.cuda.is_available() else "cpu")

def main():
    dataset_path = "data/props/tpsa.json"
    steering_strength = 0.01
    with open(dataset_path, "r") as f:
        data_dict = json.load(f)

    model, tokenizer = load_causal_lm_and_tokenizer(
        model_path="checkpoints/full_model1"
    )

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    #smiles_pairs = get_best_smiles_pairs(dataset)
    with open("smiles_pairs.json", "r") as f:
        smiles_pairs = json.load(f)["pairs"]  

    icv = pair_based_icv(smiles_pairs, model, tokenizer)

    config_steering(icv, steering_strength, model)
    
    smiles = generate_smiles(
        "C",
        1,
        model,
        tokenizer,
    )
    
    tpsas = []
    s = []
    for smi in smiles:
        tpsa = calc_tpsa(smi)

        if tpsa and tpsa > min(data_dict.values()) and tpsa < max(data_dict.values()):
            tpsas.append(tpsa)
            s.append(smi)
    sns.kdeplot(tpsas, bw_adjust=0.5)
    plt.savefig("dist.png")

    print(list(zip(s, tpsas)))
if __name__ == "__main__":
    main()