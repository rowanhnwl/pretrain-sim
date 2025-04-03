from src.latent.latent import cluster_targets_and_refs
from src.utils import split_property_dataset
from src.sim_pretraining.chemfm import LlamaForSimPred

import torch
import json

device = ("cuda" if torch.cuda.is_available() else "cpu")

def main():
    dataset_path = "data/props/tpsa.json"
    
    target = 100.0
    ref = 40.0
    tolerance = 5.0

    with open(dataset_path, "r") as f:
        data_dict = json.load(f)

    model = LlamaForSimPred(model_path="checkpoints/5M-best-tpsa/")
    model.model.to(device)
    tokenizer = model.tokenizer

    refs, targets = split_property_dataset(data_dict, target, ref, tolerance)

    cluster_targets_and_refs(
        targets=targets,
        refs=refs,
        model=model,
        tokenizer=tokenizer
    )

if __name__ == "__main__":
    main()