import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import json

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

device = ("cuda" if torch.cuda.is_available() else "cpu")

def get_final_hidden_state(smiles, model, tokenizer):
    tokenized_smiles = tokenizer(smiles, return_tensors="pt").to(device)
    del tokenized_smiles["token_type_ids"]

    with torch.no_grad():
        fwd = model.model(**tokenized_smiles)
    
    h = fwd.last_hidden_state[:, -1, :]
    return h

def get_representation_dict(targets, refs, model, tokenizer):
    rep_dict = {
        "targets": [],
        "refs": []
    }

    for t in tqdm(targets, desc="Getting target reps"):
        rep_dict["targets"].append(get_final_hidden_state(t, model, tokenizer).to("cpu"))

    for r in tqdm(refs, desc="Getting reference reps"):
        rep_dict["refs"].append(get_final_hidden_state(r, model, tokenizer).to("cpu"))

    return rep_dict

def cluster_targets_and_refs(targets, refs, model, tokenizer, save_path="clusters"):
    kmeans_model = KMeans(n_clusters=2, random_state=42)

    rep_dict = get_representation_dict(targets, refs, model, tokenizer)
    total_reps = rep_dict["targets"] + rep_dict["refs"]
    n_targets = len(rep_dict["targets"])

    M_reps = torch.stack(total_reps).squeeze(1)
    M_reps_scaled = StandardScaler().fit_transform(M_reps)

    cluster_labels = kmeans_model.fit_predict(M_reps_scaled)
    
    cluster_results_dict = {
        "targets": {
            "0": 0,
            "1": 0,
        },
        "refs": {
            "0": 0,
            "1": 0,
        }
    }

    for i, l in enumerate(cluster_labels):
        if i < n_targets:
            cluster_results_dict["targets"][str(int(l))] += 1
        else:
            cluster_results_dict["refs"][str(int(l))] += 1

    t_c0_to_c1 = cluster_results_dict["targets"]["0"] / cluster_results_dict["targets"]["1"]
    r_c0_to_c1 = cluster_results_dict["refs"]["0"] / cluster_results_dict["refs"]["1"]

    if t_c0_to_c1 > r_c0_to_c1:
        target_cluster_id = 0
        ref_cluster_id = 1
    else:
        target_cluster_id = 1
        ref_cluster_id = 0

    target_cluster_mean = torch.zeros(total_reps[0].shape)
    ref_cluster_mean = torch.zeros(total_reps[0].shape)

    for i, l in enumerate(cluster_labels):
        if i < n_targets and l == target_cluster_id:
            target_cluster_mean += total_reps[i]
        elif i >= n_targets and l == ref_cluster_id:
            ref_cluster_mean += total_reps[i]

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "clusters.json"), "w") as f:
        json.dump(cluster_results_dict, f, indent=3)

    return ref_cluster_mean, target_cluster_mean
