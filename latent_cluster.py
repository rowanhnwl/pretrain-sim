from src.generation.latent import *
from src.generation.causal import *
from src.utils import split_property_dataset
from src.sim_pretraining.chemfm import LlamaForSimPred
from src.prop_pretraining.chemfm import LlamaForPropPred

import torch
import json

device = ("cuda" if torch.cuda.is_available() else "cpu")

def main():
    dataset_path = "data/props/tpsa.json"
    
    target = 0.0
    ref = -3.0
    tolerance = 0.2

    steering_strength = 0.1

    with open(dataset_path, "r") as f:
        data_dict = json.load(f)

    model = LlamaForPropPred(model_path="checkpoints/5M-best-tpsa")
    model.model.to(device)
    tokenizer = model.tokenizer

    refs, targets = split_property_dataset(data_dict, target, ref, tolerance)
    
    means, cscores = all_layer_cluster_means(targets, refs, model, tokenizer, save_path="pt_clusters")

    print(cscores)

    #add_model_steering(targets, refs, steering_strength, model, tokenizer)

if __name__ == "__main__":
    main()