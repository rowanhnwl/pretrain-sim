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

def get_all_hidden_states(smiles, model, tokenizer):
    smiles_tokenized = tokenizer(smiles, return_tensors="pt").to(device)
    del smiles_tokenized["token_type_ids"]

    with torch.no_grad():
        fwd_pass = model.model(**smiles_tokenized, output_hidden_states=True)

    hidden_states = fwd_pass.hidden_states

    final_hidden_states_list = []

    for hs in hidden_states[1:]: # No embedding layer
        final_hidden_states_list.append(hs[:, -1, :].to("cpu"))

    return final_hidden_states_list

def get_representation_dict(targets, refs, model, tokenizer): # CHANGE TO INCORPORATE LAYERS
    rep_dict = {
        k: {
            "targets": [],
            "refs": []
        } for k in range(len(model.model.layers))
    }

    for t in tqdm(targets, desc="Getting target states"):
        all_hidden_states = get_all_hidden_states(t, model, tokenizer)
        
        for i, hs in enumerate(all_hidden_states):
            rep_dict[i]["targets"].append(hs)

    for r in tqdm(refs, desc="Getting reference states"):
        all_hidden_states = get_all_hidden_states(r, model, tokenizer)
        
        for i, hs in enumerate(all_hidden_states):
            rep_dict[i]["refs"].append(hs)

    return rep_dict

def all_layer_cluster_means(targets, refs, model, tokenizer, save_path="clusters", plot=False):
    kmeans_model = KMeans(n_clusters=2, random_state=42)
    hidden_states_dict = get_representation_dict(targets, refs, model, tokenizer)

    cluster_scores = []

    def cluster_targets_and_refs(rep_dict, k):
        
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

        n_targets_0, n_targets_1 = cluster_results_dict["targets"]["0"], cluster_results_dict["targets"]["1"]
        n_refs_0, n_refs_1 = cluster_results_dict["refs"]["0"], cluster_results_dict["refs"]["1"]
        t_c0_to_c1 = (n_targets_0 / n_targets_1) if n_targets_1 != 0 else 1.0
        r_c0_to_c1 = (n_refs_0 / n_refs_1) if n_refs_1 != 0 else 1.0

        if t_c0_to_c1 > r_c0_to_c1:
            target_cluster_id = 0
            ref_cluster_id = 1
        else:
            target_cluster_id = 1
            ref_cluster_id = 0

        score = cluster_score(target_cluster_id, cluster_results_dict)
        cluster_scores.append(score)

        target_cluster_mean = torch.zeros(total_reps[0].shape)
        ref_cluster_mean = torch.zeros(total_reps[0].shape)

        for i, l in enumerate(cluster_labels):
            if i < n_targets and l == target_cluster_id:
                target_cluster_mean += total_reps[i]
            elif i >= n_targets and l == ref_cluster_id:
                ref_cluster_mean += total_reps[i]

        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, f"layer_{k}_clusters.json"), "w") as f:
            json.dump(cluster_results_dict, f, indent=3)

        return (ref_cluster_mean, target_cluster_mean)
    
    ref_target_means = []
    for k in range(len(model.model.layers)):
        if plot:
            view_refs_targets_2d(hidden_states_dict[k]["refs"], hidden_states_dict[k]["targets"], k)
        ref_target_mean_k = cluster_targets_and_refs(hidden_states_dict[k], k)
        ref_target_means.append(ref_target_mean_k)

    return ref_target_means, cluster_scores

def cluster_score(target_cluster_id, cluster_dict):
    targets_in_c0 = cluster_dict["targets"]["0"]
    targets_in_c1 = cluster_dict["targets"]["1"]

    refs_in_c0 = cluster_dict["refs"]["0"]
    refs_in_c1 = cluster_dict["refs"]["1"]

    c0_population = targets_in_c0 + refs_in_c0
    c1_population = targets_in_c1 + refs_in_c1

    if target_cluster_id == 0:
        c0_in = targets_in_c0
        c1_in = refs_in_c1
    else:
        c0_in = refs_in_c0
        c1_in = targets_in_c1

    score = ((1/2) * (c0_in / c0_population + c1_in / c1_population))**2

    return score

def config_latent_hook(steering_vec, steering_strength):
    def latent_hook_k(module, input, output):
        hidden_states_k = output[0]

        hidden_states_k = torch.add(hidden_states_k, steering_vec * steering_strength)

        return (hidden_states_k,)
    
    return latent_hook_k

def add_latent_hooks(model, ref_target_means, cscores, steering_strength):
    for k, layer in enumerate(model.model.layers):
        if cscores[k] >= 0.0:
            ref_mean_k, target_mean_k = ref_target_means[k]
            steering_vec = (target_mean_k - ref_mean_k).to(device)

            hook_for_layer_k = config_latent_hook(steering_vec, steering_strength)

            layer.register_forward_hook(hook_for_layer_k)

def add_model_steering(targets, refs, steering_strength, model, tokenizer):
    ref_target_means, cscores = all_layer_cluster_means(
        targets,
        refs,
        model,
        tokenizer
    )

    add_latent_hooks(
        model,
        ref_target_means,
        cscores,
        steering_strength
    )

# PCA FUNCTIONS FOR PLOTTING
def pca(layer_k_data_list, N):
    layer_k_data = torch.stack(layer_k_data_list)
    mean_vector = layer_k_data.mean(dim=0, keepdim=True)

    mean_0_data = layer_k_data - mean_vector
    mean_0_data = mean_0_data.squeeze(1)

    k_cov = torch.matmul(mean_0_data.T, mean_0_data) / (len(layer_k_data) - 1)
    eigvals, eigvecs = torch.linalg.eigh(k_cov)

    sorted_inds = torch.argsort(eigvals, descending=True)
    sorted_eigvals = eigvals[sorted_inds]
    sorted_eigvecs = eigvecs[:, sorted_inds]

    return sorted_eigvals[:N], sorted_eigvecs[:N]

def down_proj(layer_k_set_list, pcs):
    layer_k_mat = torch.stack(layer_k_set_list)
    layer_k_mean = layer_k_mat.mean(dim=0, keepdim=True)

    proj = torch.matmul((layer_k_mat - layer_k_mean), pcs.T)

    return proj

def view_refs_targets_2d(layer_k_ref_list, layer_k_target_list, layer, save_path="clusters"):
    _, pcs = pca(layer_k_ref_list + layer_k_target_list, 2)

    refs_proj = down_proj(layer_k_ref_list, pcs).squeeze(1).to("cpu")
    targets_proj = down_proj(layer_k_target_list, pcs).squeeze(1).to("cpu")

    plt.scatter(refs_proj[:, 0], refs_proj[:, 1], color='b')
    plt.scatter(targets_proj[:, 0], targets_proj[:, 1], color='r')
    plt.savefig(f"{save_path}/layer_{layer}_clusters.png")
    plt.cla()