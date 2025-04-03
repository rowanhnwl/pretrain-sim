from transformers import AutoTokenizer, LlamaModel

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

device = ("cuda" if torch.cuda.is_available() else "cpu")

class LlamaForSimPred(nn.Module):
    def __init__(self, model_path, hf_path="ChemFM/ChemFM-1B", embed_dim=2048):
        super().__init__()

        self.model = LlamaModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_path)

        self.tokenizer.add_special_tokens({
            'pad_token': '<pad>'
        })
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.proj_to_3d = nn.Linear(embed_dim, 3)

        print(f"Loaded from {model_path}")

    def forward(self, smi1_input_ids, smi1_attn_mask, smi2_input_ids, smi2_attn_mask):
        batch_size = smi1_input_ids.shape[0]
        
        fwd1 = self.model(input_ids=smi1_input_ids, attention_mask=smi1_attn_mask)
        fwd2 = self.model(input_ids=smi2_input_ids, attention_mask=smi2_attn_mask)

        h1 = fwd1.last_hidden_state[torch.arange(batch_size), smi1_attn_mask.sum(dim=1) - 1, :]
        h2 = fwd2.last_hidden_state[torch.arange(batch_size), smi2_attn_mask.sum(dim=1) - 1, :]

        h1_3d = self.proj_to_3d(h1).squeeze(1)
        h2_3d = self.proj_to_3d(h2).squeeze(1)

        sim_pred = F.cosine_similarity(h1_3d, h2_3d)

        return sim_pred

def train_for_sim_pred(model, train_dataloader, valid_dataloader, optimizer, epochs, ckpt_path):

    model.to(device)
    best_valid_loss = float("inf")

    criterion = nn.L1Loss()
    
    for epoch in range(epochs):

        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            sim_pred = model(
                smi1_input_ids=batch["smi1_input_ids"],
                smi1_attn_mask=batch["smi1_attn_mask"],
                smi2_input_ids=batch["smi2_input_ids"],
                smi2_attn_mask=batch["smi2_attn_mask"]
            )

            loss = criterion(sim_pred, batch["sim"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        mean_epoch_loss = epoch_loss / len(train_dataloader)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc=f"Validation for epoch {epoch+1}/{epochs}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                sim_pred = model(
                    smi1_input_ids=batch["smi1_input_ids"],
                    smi1_attn_mask=batch["smi1_attn_mask"],
                    smi2_input_ids=batch["smi2_input_ids"],
                    smi2_attn_mask=batch["smi2_attn_mask"]
                )

                loss = criterion(sim_pred, batch["sim"])

                valid_loss += loss.item()
        mean_valid_loss = valid_loss / len(valid_dataloader)

        print(f"Epoch {epoch+1}/{epochs}: loss={mean_epoch_loss} | valid_loss={mean_valid_loss}")

        if mean_valid_loss < best_valid_loss:
            best_valid_loss = mean_valid_loss

            print(f"Saving at {ckpt_path}")
            model.model.save_pretrained(ckpt_path)

def test_sim_pred(model, test_dataloader):
    pred_sims = []
    real_sims = []

    model.to(device)
    criterion = nn.L1Loss()
    
    with torch.no_grad():
        model.eval()

        test_loss = 0.0
        for batch in tqdm(test_dataloader, desc="Testing"):
            batch = {k: v.to(device) for k, v in batch.items()}
            sim_pred = model(
                smi1_input_ids=batch["smi1_input_ids"],
                smi1_attn_mask=batch["smi1_attn_mask"],
                smi2_input_ids=batch["smi2_input_ids"],
                smi2_attn_mask=batch["smi2_attn_mask"]
            )

            loss = criterion(sim_pred, batch["sim"])

            pred_sims += sim_pred.to("cpu").tolist()
            real_sims += batch["sim"].to("cpu").tolist()

            test_loss += loss

    mean_test_loss = test_loss / len(test_dataloader)

    print(f"test_loss={mean_test_loss}")

    return pred_sims, real_sims