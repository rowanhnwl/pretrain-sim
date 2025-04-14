from src.generation.latent import *
from src.generation.causal import *
from src.utils import split_property_dataset, calc_tpsa

from src.prop_pretraining.chemfm import LlamaForPropPred

import torch
import json

import seaborn as sns

device = ("cuda" if torch.cuda.is_available() else "cpu")

def main():
    dataset_path = "data/props/tpsa.json"
    
    r_range = (40.0, 50.0)
    t_range = (80.0, 90.0)

    steering_strength = 0.01

    with open(dataset_path, "r") as f:
        data_dict = json.load(f)

    model, tokenizer = load_causal_lm_and_tokenizer(
        model_path="checkpoints/full_model1"
    )

    refs, targets = split_property_dataset(data_dict, t_range, r_range)

    fig, ax = plt.subplots(22, 1)

    pcs_list = full_dataset_layers(list(data_dict.keys()), list(data_dict.values()), model, tokenizer, ax)

    add_model_steering(targets, refs, steering_strength, model, tokenizer, pcs_list, ax)
    smiles = generate_smiles(
        "C",
        100,
        model,
        tokenizer,
    )

    # for p, k in enumerate(plts_pcs):
    #     plt_obj = p[0]
    #     plt_obj.savefig(f"layer_{k}.png")
    
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