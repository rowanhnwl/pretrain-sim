from src.generation.latent import *
from src.generation.causal import *
from src.utils import *
from src.generation.eval import eval_prop

from tqdm import tqdm
import torch
import json

import seaborn as sns
import os
import pandas as pd

from itertools import product

device = ("cuda" if torch.cuda.is_available() else "cpu")

path_dict = {
    "tpsa": {
        "dataset": "data/props/tpsa.json",
        "pairs": "data/pairs/tpsa_smiles_pairs.json",
        "states": "data/states/tpsa_states.pickle"
    },
    "xlogp": {
        "dataset": "data/props/xlogp.json",
        "pairs": "data/pairs/xlogp_smiles_pairs.json",
        "states": "data/states/xlogp_states.pickle"
    },
    "molecular_weight": {
        "dataset": "data/props/molecular_weight.json",
        "pairs": "data/pairs/molecular_weight_smiles_pairs.json",
        "states": "data/states/molecular_weight_states.pickle"
    },
    "caco2_permeability": {
        "dataset": "data/props/caco2_permeability.json",
        "pairs": "data/pairs/caco2_permeability_smiles_pairs.json",
        "states": "data/states/caco2_permeability_states.pickle"
    },
    "acute_toxicity": {
        "dataset": "data/props/acute_toxicity.json",
        "pairs": "data/pairs/acute_toxicity_smiles_pairs.json",
        "states": "data/states/acute_toxicity_states.pickle"
    },
    "solubility": {
        "dataset": "data/props/solubility.json",
        "pairs": "data/pairs/solubility_smiles_pairs.json",
        "states": "data/states/solubility_states.pickle"
    },
    "lipophilicity": {
        "dataset": "data/props/lipophilicity.json",
        "pairs": "data/pairs/lipophilicity_smiles_pairs.json",
        "states": "data/states/lipophilicity_states.pickle"
    },
    "rotatable_bond_count": {
        "dataset": "data/props/rotatable_bond_count.json",
        "pairs": "data/pairs/rotatable_bond_count_smiles_pairs.json",
        "states": "data/states/rotatable_bond_count_states.pickle"
    }
}

def generate_guided_molecules(props, n, n_clusters, model, tokenizer, output_path):
    
    prop_icvs = []
    
    for prop, strength in props:
        pairs_path = path_dict[prop]["pairs"]
        try:
            states_path = path_dict[prop]["states"]
        except:
            states_path = None

        with open(pairs_path, "r") as f:
            smiles_pairs = json.load(f)["pairs"]

        icv = pair_based_icv(smiles_pairs, model, tokenizer, n_clusters=n_clusters, states_path=states_path)
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

        with open(path_dict[prop]["dataset"], "r") as f:
            dataset_vals = list(json.load(f).values())
        dataset_min, dataset_max = min(dataset_vals), max(dataset_vals)

        eval_dict = eval_prop(smiles, prop)
        out_dict[prop] = eval_dict
        vals[prop] = [x for x in list(eval_dict.values()) if x >= dataset_min and x <= dataset_max]

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "molecules.json"), "w") as f:
        json.dump(out_dict, f, indent=3)

    if len(list(vals.keys())) == 1:
        sns.kdeplot(list(vals.values())[0])
    else:
        df = pd.DataFrame(vals)
        props = list(vals.keys())
        sns.jointplot(data=df, x=props[0], y=props[1], kind="kde")
    plt.savefig(os.path.join(output_path, "dist.png"))
    plt.cla()

def grid_search_single():
    props_list = [
        "caco2_permeability"
    ]

    ss_list = [-0.1, -0.2, -0.3, -0.5, -0.7, 0.1, 0.2, 0.3, 0.5, 0.7]
    clusters_list = [5, 10, 15, 25, 50, 75]

    configs = product(
        props_list,
        ss_list,
        clusters_list
    )

    model, tokenizer = load_causal_lm_and_tokenizer(
        model_path="checkpoints/final",
        hf_path="ChemFM/ChemFM-3B"
    )

    n = 250

    for (prop, ss, n_clusters) in tqdm(configs):
        outpath = f"molecules/{prop}_{ss}_{n_clusters}"

        generate_guided_molecules(
            [(prop, ss)],
            n,
            n_clusters,
            model,
            tokenizer,
            output_path=outpath
        )

def grid_search_dual():
    dual_props_list = [
        [
            "solubility",
            "caco2_permeability"
        ],
        [
            "xlogp",
            "caco2_permeability"
        ],
        [
            "acute_toxicity",
            "caco2_permeability"
        ],
        [
            "rotatable_bond_count",
            "caco2_permeability"
        ],
        [
            "tpsa",
            "caco2_permeability"
        ],
        [
            "lipophilicity",
            "caco2_permeability"
        ],
        [
            "tpsa",
            "lipophilicity"
        ],
        [
            "solubility",
            "lipophilicity"
        ],
        [
            "xlogp",
            "lipophilicity"
        ],
        [
            "rotatable_bond_count",
            "lipophilicity"
        ]
    ]

def main():

    # grid_search_single()

    props = [
        ("solubility", 0.7)
    ]

    n = 10
    n_clusters = 10

    model, tokenizer = load_causal_lm_and_tokenizer(
        model_path="checkpoints/final",
        hf_path="ChemFM/ChemFM-3B"
    )

    generate_guided_molecules(props, n, n_clusters, model, tokenizer, output_path="molecules/solubility_high")

if __name__ == "__main__":
    main()