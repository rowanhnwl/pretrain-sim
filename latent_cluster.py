from src.generation.latent import *
from src.generation.causal import *
from src.utils import split_property_dataset

from src.prop_pretraining.chemfm import LlamaForPropPred

import torch
import json

device = ("cuda" if torch.cuda.is_available() else "cpu")

def main():
    dataset_path = "data/props/tpsa.json"
    
    target = 100.0
    ref = 40.0
    tolerance = 5.0

    steering_strength = 0.1

    with open(dataset_path, "r") as f:
        data_dict = json.load(f)

    model, tokenizer = load_causal_lm_and_tokenizer(
        model_path="output/checkpoint-26565"
    )

    refs, targets = split_property_dataset(data_dict, target, ref, tolerance)
    
    #means, cscores = all_layer_cluster_means(targets, refs, model, tokenizer, save_path="pt_clusters")
    add_model_steering(targets, refs, steering_strength, model, tokenizer)

    smiles = generate_smiles(
        "C",
        100,
        model,
        tokenizer,
    )

    print(smiles)

if __name__ == "__main__":
    main()