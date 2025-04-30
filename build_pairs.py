import pickle
import json

from src.preprocess import get_best_smiles_pairs_latent
from src.generation.causal import load_causal_lm_and_tokenizer
from src.generation.latent import get_representation_dict

dataset_path = "data/props/acute_toxicity.json"
pairs_path = "data/pairs/acute_toxicity_smiles_pairs.json"
states_path = "data/states/acute_toxicity_states.pickle"

with open(dataset_path, "r") as f:
                dataset = json.load(f)

model, tokenizer = load_causal_lm_and_tokenizer(
        model_path="checkpoints/final",
        hf_path="ChemFM/ChemFM-3B"
    )

n_pairs = 100000
sample_rate = 0.25
smiles_pairs = get_best_smiles_pairs_latent(dataset, sample_rate, n_pairs, model, tokenizer)
with open(pairs_path, "w") as f:
    json.dump({"pairs": smiles_pairs}, f, indent=3)

all_smiles = []
for pair in smiles_pairs:
    all_smiles.append(pair[0])
    all_smiles.append(pair[1])

all_smiles = list(set(all_smiles))

rep_dict = get_representation_dict(all_smiles, model, tokenizer)
with open(states_path, "wb") as f:
    pickle.dump(rep_dict, f)