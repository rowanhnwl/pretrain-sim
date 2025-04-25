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

def get_representation_dict(smiles, model, tokenizer): # CHANGE TO INCORPORATE LAYERS
    rep_dict = {
        k: {} for k in range(len(model.model.layers))
    }

    for t in tqdm(smiles, desc="Getting states"):
        all_hidden_states = get_all_hidden_states(t, model, tokenizer)
        
        for i, hs in enumerate(all_hidden_states):
            rep_dict[i][t] = hs

    return rep_dict

def sort_rep_dict(rep_dict, prop_list):
    for k in rep_dict.keys():
        pair_list = list(zip(rep_dict[k], prop_list))
        sorted_pair_list = sorted(pair_list, key=lambda x: x[1])

        rep_dict[k] = deepcopy(sorted_pair_list)

    return rep_dict

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

def config_latent_hook(icv):
    def latent_hook_k(module, input, output):
        hidden_states_k = output[0]

        #adapted_lambdas = get_adapted_lambda(icv, hidden_states_k, steering_strength)
        #new_hidden_states_k = torch.add(icv.view(1, 1, -1) * adapted_lambdas.unsqueeze(-1), hidden_states_k)

        new_hidden_states_k = torch.add(icv.view(1, 1, -1), hidden_states_k)

        norm = torch.norm(hidden_states_k, p=2, dim=2, keepdim=True)
        new_norm = torch.norm(new_hidden_states_k, p=2, dim=2, keepdim=True)

        hidden_states_k = torch.mul(new_hidden_states_k, (norm / new_norm))

        return (hidden_states_k,)
    
    return latent_hook_k

def config_steering(icv, model):

    hook_handles = []

    for k, layer in enumerate(model.model.layers):

        hook_for_layer_k = config_latent_hook(
            icv[k].to(device)
        )

        handle = layer.register_forward_hook(hook_for_layer_k)
        hook_handles.append(handle)

    return hook_handles

# Property value based approach
def icv_prop_grad(h, h_list_sorted, alpha, p_d, sample_rate=0.1):
    p_hat = h_prop_pred(h, h_list_sorted, sample_rate)
    p_dt = alpha * p_d + (1 - alpha) * p_hat
    
    p_dt_list = p_dt.tolist()

    ref_samples_lists = [torch.stack(closest_to_pdt(h_list_sorted, int(sample_rate * len(h_list_sorted)), pdt)) for pdt in p_dt_list]

    S_dt = torch.stack(ref_samples_lists).squeeze(2).to(device)
    diff_mat = S_dt - h.unsqueeze(1)

    h_icv = (alpha / len(ref_samples_lists)) * torch.sum(diff_mat, dim=1)

    return h_icv

def h_prop_pred(h, h_list_sorted, sample_rate=0.1):

    n_sample = int(len(h_list_sorted) * sample_rate)
    samples = random.sample(h_list_sorted, n_sample)

    h_samples = [s[0] for s in samples]
    p_samples = torch.tensor([s[1] for s in samples]).to(device)

    h_mat = torch.stack(h_samples).to(device)
    
    dist_mat = torch.norm(h_mat - h, p=2, dim=2)
    dist_sum = torch.sum(dist_mat, dim=0)

    p_hat = (1 / dist_sum) * torch.sum(p_samples.view(-1, 1) / dist_mat)

    return p_hat

def closest_to_pdt(sorted_layer_pairs, n, p_dt):
    n_layer_pairs = len(sorted_layer_pairs)
    left = 0
    right = n_layer_pairs - 1

    best_ind = 0
    best_diff = float('inf')

    while left <= right:
        mid = (left + right) // 2
        mid_val = sorted_layer_pairs[mid][1]
        diff = abs(mid_val - p_dt)

        if diff < best_diff:
            best_diff = diff
            best_ind = mid

        if mid_val < p_dt:
            left = mid + 1
        elif mid_val > p_dt:
            right = mid - 1
        else:
            best_ind = mid
            break

    n_closest = [sorted_layer_pairs[best_ind]]
    right = best_ind + 1
    left = best_ind - 1
    while len(n_closest) < n:

        remaining = n - len(n_closest)

        if right > n_layer_pairs - 1:
            n_closest += sorted_layer_pairs[best_ind - remaining:best_ind]
            break
        elif left < 0:
            n_closest += sorted_layer_pairs[best_ind + 1:best_ind + remaining + 1]
            break

        right_diff = abs(p_dt - sorted_layer_pairs[right][1])
        left_diff = abs(p_dt - sorted_layer_pairs[left][1])

        if right_diff < left_diff:
            n_closest.append(sorted_layer_pairs[right])
            right += 1
        else:
            n_closest.append(sorted_layer_pairs[left])
            left -= 1

    n_closest_list = [nc[0] for nc in n_closest]

    return n_closest_list

# Pair based approach
def pair_based_icv(smiles_pairs, model, tokenizer):
    smiles_list = [x[0] for x in smiles_pairs]
    smiles_list += [x[1] for x in smiles_pairs]

    val_diffs = torch.tensor([x[3] for x in smiles_pairs])
    val_diff_sum = val_diffs.sum()

    smiles_list = list(set(smiles_list))

    rep_dict = get_representation_dict(smiles_list, model, tokenizer)

    icv_list = [torch.zeros((1, model.config.hidden_size)) for k in range(len(model.model.layers))]
    
    for pair in smiles_pairs:
        smi1, smi2, sim, pval = pair

        pval_adj = (pval - val_diffs.min()) / (val_diffs.max() - val_diffs.min())

        for k in range(len(model.model.layers)):
            smi1_layer_k = rep_dict[k][smi1]
            smi2_layer_k = rep_dict[k][smi2]

            diff_vec = smi1_layer_k - smi2_layer_k # Targetting larger prop vals
            #diffs.append(torch.norm(diff_vec, p=2))
            alpha = 1 / torch.norm(diff_vec, p=2)

            icv_list[k] += diff_vec #* ((1.0 - (pval / val_diff_sum))/(val_diffs.max()-val_diffs.min()))

    icv_list = [icv / len(smiles_pairs) for icv in icv_list]

    return icv_list