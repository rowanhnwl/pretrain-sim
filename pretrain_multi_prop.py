from src.utils import *
from src.prop_pretraining.data import *
from src.prop_pretraining.chemfm import *

from torch.utils.data import DataLoader, random_split
import torch.optim as optim

def main():
    model_path = "ChemFM/ChemFM-1B"
    ckpt_path = "checkpoints/multi"

    n_samples = 10000
    valid_ratio = 0.20
    batch_size = 16
    lr = 1e-5
    epochs = 10

    zinc250k_smiles = load_zinc250k_all()
    os.makedirs(ckpt_path, exist_ok=True)

    model = LlamaForMultiPropPred(model_path=model_path)

    pair_sim_dataset = PropDataset(
        master_data_list=zinc250k_smiles,
        n_samples=n_samples,
        tokenizer=model.tokenizer,
        n_props=3
    )

    valid_size = int(valid_ratio * len(pair_sim_dataset))
    train_size = len(pair_sim_dataset) - valid_size

    train_dataset, valid_dataset = random_split(pair_sim_dataset, [train_size, valid_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_batch)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_for_multi_prop_pred(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        epochs=epochs,
        ckpt_path=ckpt_path
    )

if __name__ == "__main__":
    main()