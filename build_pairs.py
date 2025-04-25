import json
from preprocess import get_best_smiles_pairs

dataset_path = "data/props/caco2_permeability.json"
pairs_path = "data/pairs/caco2_permeability_smiles_pairs.json"

with open(dataset_path, "r") as f:
                dataset = json.load(f)

smiles_pairs = get_best_smiles_pairs(dataset, sample_rate=1.0, sim_thresh=0.97)
with open(pairs_path, "w") as f:
    json.dump({"pairs": smiles_pairs}, f, indent=3)