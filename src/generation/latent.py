import torch
import torch.nn.functional as F

from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import json

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from copy import deepcopy
import math
import random
from heapq import nsmallest

device = ("cuda" if torch.cuda.is_available() else "cpu")

EOS_TOKEN_ID = 265

def cat_for_eos(tokenized):
    tokenized["input_ids"] = torch.cat(
        (
            tokenized["input_ids"],
            torch.tensor([[EOS_TOKEN_ID]], dtype=torch.long, device=device)
        ),
        dim=1
    )
    tokenized["attention_mask"] = torch.cat(
        (
            tokenized["attention_mask"],
            torch.tensor([[1]], dtype=torch.long, device=device)
        ),
        dim=1
    )

def get_all_hidden_states(smiles, model, tokenizer):
    smiles_tokenized = tokenizer(smiles, return_tensors="pt").to(device)
    del smiles_tokenized["token_type_ids"]
    cat_for_eos(smiles_tokenized)

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

def get_adapted_lambda(h_icv, hidden_states_k, lam):
    h_icv_expanded = h_icv.squeeze(1).view(1, 1, -1)
    cos_sim_mat = F.cosine_similarity(hidden_states_k, -h_icv_expanded, dim=-1)

    cos_sim_clamped = torch.clamp(cos_sim_mat, min=0.0)

    adapted_lambdas = lam * (1 + cos_sim_clamped)

    return adapted_lambdas

def save_ax_as_png(ax, filename):
    # Create new figure and copy the Axes into it
    fig = plt.figure()
    new_ax = fig.add_subplot(111)

    # Copy scatter plots (collections)
    for col in ax.collections:
        offsets = col.get_offsets()
        facecolors = col.get_facecolors()
        sizes = col.get_sizes()
        if len(offsets) > 0:
            new_ax.scatter(
                offsets[:, 0], offsets[:, 1],
                c=facecolors if len(facecolors) > 0 else None,
                s=sizes if len(sizes) > 0 else None,
                label=col.get_label()
            )

    # Save and clean up
    fig.savefig(filename)
    plt.close(fig)

def config_latent_hook(steering_vec, steering_strength, k, plt_obj=None, layer_k_mean=None, pcs=None):
    def latent_hook_k(module, input, output):
        hidden_states_k = output[0]
        
        adapted_lambdas = get_adapted_lambda(steering_vec, hidden_states_k, steering_strength)
        new_hidden_states_k = torch.add(steering_vec.view(1, 1, -1) * adapted_lambdas.unsqueeze(-1), hidden_states_k)

        norm = torch.norm(hidden_states_k, p=2, dim=2, keepdim=True)
        new_norm = torch.norm(new_hidden_states_k, p=2, dim=2, keepdim=True)

        hidden_states_k = torch.mul(new_hidden_states_k, (norm / new_norm))

        last_hidden_state_k = hidden_states_k[:, -1, :]

        # t = hidden_states_k.shape[1]

        # lhs_k_proj = down_proj_single_h(last_hidden_state_k.to("cpu"), layer_k_mean, pcs)

        # plt_obj.scatter(lhs_k_proj[:, :, 0], lhs_k_proj[:, :, 1], color='r')
        
        # save_ax_as_png(plt_obj, f"layer_plots/layer_{k}_token_{t}.png")

        return (hidden_states_k,)
    
    return latent_hook_k

def add_latent_hooks(model, rep_dict, steering_strength, pcs_list, ax):
    for k, layer in enumerate(model.model.layers):
        h_icv = unpaired_h_icv(rep_dict, k).to(device)
        pcs_k, lkm = pcs_list[k]
        hook_for_layer_k = config_latent_hook(h_icv, steering_strength, k, ax[k], lkm, pcs_k)

        layer.register_forward_hook(hook_for_layer_k)

def add_latent_hooks_by_mean(model, ref_target_means, cscores, steering_strength, pcs_list, ax):
    for k, layer in enumerate(model.model.layers):
        ref_mean_k, target_mean_k = ref_target_means[k]
        h_icv = (target_mean_k - ref_mean_k).to(device)
        pcs_k, lkm = pcs_list[k]
        hook_for_layer_k = config_latent_hook(h_icv, steering_strength, k, ax[k], lkm, pcs_k)

        layer.register_forward_hook(hook_for_layer_k)

def get_P_y(h_y_dot, h_x_sum):

    P_y = h_y_dot / (h_y_dot + h_x_sum)

    return P_y

def get_P_x(h_y_dot, h_x_dot, h_x_sum):

    P_x = h_x_dot / (h_y_dot + h_x_sum)

    return P_x

def unpaired_h_icv(rep_dict, k):
    target_reps = rep_dict[k]["targets"]
    ref_reps = rep_dict[k]["refs"]

    exp_ref_dot_list = []
    good_ref_reps = []
    for h_x in ref_reps:
        try:
            exp_h_x_dot = math.exp(torch.dot(h_x.squeeze(0), h_x.squeeze(0)))
            exp_ref_dot_list.append(exp_h_x_dot)
            good_ref_reps.append(h_x)
        except:
            continue

    exp_target_dot_list = []
    for h_y in target_reps:
        exp_target_dot = math.exp(torch.dot(h_y.squeeze(0), h_y.squeeze(0)))
        exp_target_dot_list.append(exp_target_dot)

    ref_sum = sum(exp_ref_dot_list)

    h_icv = sum([
        (1.0 - get_P_y(t_rep_dot, ref_sum)) * t_rep - sum([
            get_P_x(t_rep_dot, r_rep_dot, ref_sum) * r_rep for r_rep, r_rep_dot in zip(good_ref_reps, exp_ref_dot_list)
        ])
        for t_rep, t_rep_dot in zip(target_reps, exp_target_dot_list)
    ])

    return h_icv

def add_model_steering(targets, refs, steering_strength, model, tokenizer, pcs_list, ax):
    ref_target_means, cscores = all_layer_cluster_means(
        targets,
        refs,
        model,
        tokenizer
    )

    add_latent_hooks_by_mean(
        model,
        ref_target_means,
        cscores,
        steering_strength,
        pcs_list,
        ax
    )

    # rep_dict = get_representation_dict(targets, refs, model, tokenizer)

    # add_latent_hooks(
    #     model,
    #     rep_dict,
    #     steering_strength,
    #     pcs_list,
    #     ax
    # )

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

    return proj, layer_k_mean

def down_proj_single_h(h, layer_k_mean, pcs):
    proj = torch.matmul((h - layer_k_mean), pcs.T)

    return proj

def view_refs_targets_2d(layer_k_ref_list, layer_k_target_list, layer, save_path="clusters"):
    _, pcs = pca(layer_k_ref_list + layer_k_target_list, 2)

    refs_proj = down_proj(layer_k_ref_list, pcs).squeeze(1).to("cpu")
    targets_proj = down_proj(layer_k_target_list, pcs).squeeze(1).to("cpu")

    plt.scatter(refs_proj[:, 0], refs_proj[:, 1], color='b')
    plt.scatter(targets_proj[:, 0], targets_proj[:, 1], color='r')
    plt.savefig(f"{save_path}/layer_{layer}_clusters.png")
    plt.cla()

def get_full_representation_dict(smiles, model, tokenizer):
    rep_dict = {
        k: [] for k in range(len(model.model.layers))
    }

    for smi in tqdm(smiles, desc="Getting target states"):
        all_hidden_states = get_all_hidden_states(smi, model, tokenizer)
        
        for i, hs in enumerate(all_hidden_states):
            rep_dict[i].append(hs)

    return rep_dict

def view_all_2d(layer_k_list, layer, prop_vals, ax, save_path="clusters"):
    _, pcs = pca(layer_k_list, 2)

    proj, lkm = down_proj(layer_k_list, pcs)
    proj = proj.squeeze(1).to("cpu")

    ax.scatter(proj[:, 0], proj[:, 1], c=prop_vals, cmap='viridis', vmin=min(prop_vals), vmax=max(prop_vals))

    return (pcs, lkm)

def full_dataset_layers(smiles, prop_vals, model, tokenizer, ax):
    rep_dict = get_full_representation_dict(smiles, model, tokenizer)

    pcs_list = []

    for k in range(len(model.model.layers)):
        pcs_list.append(view_all_2d(rep_dict[k], k, prop_vals, ax[k]))

    return pcs_list

# Property value based approach
def icv_prop_grad(h, h_list, prop_list, alpha, p_d, sample_rate=0.2):
    p_hat = h_prop_pred(h, h_list, prop_list, sample_rate)
    p_dt = alpha * p_d + (1 - alpha) * p_hat
    
    samples = list(zip(h_list, prop_list))

    ref_samples = nsmallest(int(len(samples) * sample_rate), samples, key=lambda x: abs(x[1] - p_dt))
    S_dt = torch.stack([s[0] for s in ref_samples])

    diff_mat = S_dt - h

    h_icv = (alpha / len(ref_samples)) * torch.sum(diff_mat)

    print(h_icv)

def h_prop_pred(h, h_list, prop_list, sample_rate=0.2):

    n_sample = int(len(h_list) * sample_rate)
    samples = random.sample(list(zip(h_list, prop_list)), n_sample)

    h_samples = [s[0] for s in samples]
    p_samples = torch.tensor([s[1] for s in samples])

    h_mat = torch.stack(h_samples)
    
    dist_mat = torch.norm(h_mat - h, p=2)
    dist_sum = torch.sum(dist_mat)

    p_hat = (1 / dist_sum) * torch.sum(p_samples / dist_mat)

    return p_hat