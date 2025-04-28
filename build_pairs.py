import json
from src.preprocess import get_best_smiles_pairs_latent
from src.generation.causal import load_causal_lm_and_tokenizer

dataset_path = "data/props/tpsa.json"
pairs_path = "data/pairs/tpsa_smiles_pairs.json"

with open(dataset_path, "r") as f:
                dataset = json.load(f)

model, tokenizer = load_causal_lm_and_tokenizer(
        model_path="checkpoints/final",
        hf_path="ChemFM/ChemFM-3B"
    )

n_pairs = 500
sample_rate = 1.0
smiles_pairs = get_best_smiles_pairs_latent(dataset, sample_rate, n_pairs, model, tokenizer)
with open(pairs_path, "w") as f:
    json.dump({"pairs": smiles_pairs}, f, indent=3)