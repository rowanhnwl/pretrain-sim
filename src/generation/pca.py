import torch
from matplotlib import pyplot as plt
import tqdm

from src.generation.latent import get_all_hidden_states

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
