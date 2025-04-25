from src.generation.latent import *
from src.generation.causal import *
from src.utils import *
from src.generation.eval import eval_prop

from src.prop_pretraining.chemfm import LlamaForPropPred

from preprocess import get_best_smiles_pairs

from tqdm import tqdm
import torch
import json

import seaborn as sns
import os
import pandas as pd

device = ("cuda" if torch.cuda.is_available() else "cpu")

path_dict = {
    "tpsa": {
        "dataset": "data/props/tpsa.json",
        "pairs": "data/pairs/tpsa_smiles_pairs.json"
    },
    "xlogp": {
        "dataset": "data/props/xlogp.json",
        "pairs": "data/pairs/xlogp_smiles_pairs.json"
    },
    "molecular_weight": {
        "dataset": "data/props/molecular_weight.json",
        "pairs": "data/pairs/molecular_weight_smiles_pairs.json"
    },
    "caco2_permeability": {
        "dataset": "data/props/caco2_permeability.json",
        "pairs": "data/pairs/caco2_permeability_smiles_pairs.json"
    }
}

def generate_guided_molecules(props, n, model, tokenizer, output_path, build_pairs=False):
    
    prop_icvs = []
    
    for prop, strength in props:
        dataset_path = path_dict[prop]["dataset"]
        pairs_path = path_dict[prop]["pairs"]

        if build_pairs:

            with open(dataset_path, "r") as f:
                dataset = json.load(f)

            smiles_pairs = get_best_smiles_pairs(dataset)
            with open(pairs_path, "w") as f:
                json.dump({"pairs": smiles_pairs}, f, indent=3)

        with open(pairs_path, "r") as f:
            smiles_pairs = json.load(f)["pairs"]

        icv = pair_based_icv(smiles_pairs, model, tokenizer)
        icv = [i * strength for i in icv]
        prop_icvs.append(icv)

    summed_icv = []
    for k in range(len(prop_icvs[0])):

        layer_k_icvs = [prop_icvs[i][k] for i in range(len(prop_icvs))]
        summed_icv.append(sum(layer_k_icvs))

    handles = config_steering(summed_icv, model)

    smiles = []
    for _ in tqdm(range(n)):
        smiles += generate_smiles(
            "C",
            1,
            model,
            tokenizer,
        )   
    
    for handle in handles:
        handle.remove()
    del model

    out_dict = {}
    vals = {}
    for prop, strength in props:
        eval_dict = eval_prop(smiles, prop)
        out_dict[prop] = eval_dict
        vals[prop] = list(eval_dict.values())

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "molecules.json"), "w") as f:
        json.dump(out_dict, f, indent=3)

    if len(list(vals.keys())) == 1:
        sns.kdeplot(list(vals.values())[0], bw_adjust=0.5)
    else:
        df = pd.DataFrame(vals)
        props = list(vals.keys())
        sns.jointplot(data=df, x=props[0], y=props[1], kind="kde", bw_adjust=0.5)
    plt.savefig(os.path.join(output_path, "dist.png"))

def main():

    props = [
        ("caco2_permeability", -0.6),
        ("tpsa", 0.3)
    ]

    n = 50

    model, tokenizer = load_causal_lm_and_tokenizer(
        model_path="output/final",
        hf_path="ChemFM/ChemFM-3B"
    )

    generate_guided_molecules(props, n, model, tokenizer, output_path="molecules/caco2_permeability_low_tpsa_high")

if __name__ == "__main__":
    main()