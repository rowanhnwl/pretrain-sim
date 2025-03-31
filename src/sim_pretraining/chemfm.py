from transformers import AutoTokenizer, LlamaModel

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

device = ("cuda" if torch.cuda.is_available() else "cpu")

class LlamaForSimPred(nn.Module):
    def __init__(self, model_path):
        super().__init__()

        self.model = LlamaModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.tokenizer.add_special_tokens({
            'pad_token': '<pad>'
        })
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.resize_token_embeddings(len(self.tokenizer))

        print(f"Loaded from {model_path}")

    def forward(self, smi1_input_ids, smi1_attn_mask, smi2_input_ids, smi2_attn_mask):
        batch_size = smi1_input_ids.shape[0]
        
        fwd1 = self.model(input_ids=smi1_input_ids, attention_mask=smi1_attn_mask)
        fwd2 = self.model(input_ids=smi2_input_ids, attention_mask=smi2_attn_mask)

        h1 = fwd1.last_hidden_state[torch.arange(batch_size), smi1_attn_mask.sum(dim=1) - 1, :]
        h2 = fwd2.last_hidden_state[torch.arange(batch_size), smi2_attn_mask.sum(dim=1) - 1, :]

        sim_pred = F.cosine_similarity(h1, h2)

        return sim_pred

def train_for_sim_pred(model, train_dataloader, valid_dataloader, optimizer, epochs, ckpt_path):

    model.to(device)
    best_valid_loss = float("inf")
    
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

            loss = F.mse_loss(sim_pred, batch["sim"])

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

                loss = F.mse_loss(sim_pred, batch["sim"])

                valid_loss += loss.item()
        mean_valid_loss = valid_loss / len(valid_dataloader)

        print(f"Epoch {epoch+1}/{epochs}: loss={mean_epoch_loss} | valid_loss={mean_valid_loss}")

        if mean_valid_loss < best_valid_loss:
            best_valid_loss = mean_valid_loss

            print(f"Saving at {ckpt_path}")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': best_valid_loss
            }, ckpt_path + "/best.pt")