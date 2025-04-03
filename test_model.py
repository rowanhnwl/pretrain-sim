from src.utils import *
from src.sim_pretraining.data import *
from src.sim_pretraining.chemfm import *

import torch
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt

def main():
    model_path = "checkpoints"
    model = LlamaForSimPred(model_path)
    
    test_size = 1000

    zinc250k = load_zinc250k()
    test_dataset = TanimotoDataset(zinc250k, test_size, model.tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_batch)

    pred_sims, real_sims = test_sim_pred(
        model,
        test_dataloader
    )

    plt.hist(pred_sims, bins=100)
    plt.savefig("pred_hist.png")
    plt.cla()
    plt.hist(real_sims, bins=100)
    plt.savefig("real_hist.png")
    plt.cla()

if __name__ == "__main__":
    main()