from transformers import AutoTokenizer, LlamaModel
from peft import get_peft_model, LoraConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import os
from copy import deepcopy

device = ("cuda" if torch.cuda.is_available() else "cpu")

class LlamaForPropPred(nn.Module):
    def __init__(self, model_path, hf_path="ChemFM/ChemFM-1B", embed_dim=2048, lora=False):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(hf_path)
        self.embed_dim = embed_dim

        if lora and model_path is not None:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                bias="none"
            )

            base_model = LlamaModel.from_pretrained(model_path)
            self.model = get_peft_model(base_model, lora_config)
        elif model_path is not None:
            self.model = LlamaModel.from_pretrained(model_path)

        if model_path is not None:
            self.tokenizer.add_special_tokens({
                'pad_token': '<pad>'
            })
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.resize_token_embeddings(len(self.tokenizer))

            print(f"Loaded from {model_path}")

        self.pred_nn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1)
        )

    def forward(self, smi_input_ids, smi_attn_mask):
        batch_size = smi_input_ids.shape[0]
        
        fwd = self.model(input_ids=smi_input_ids, attention_mask=smi_attn_mask)
        h = fwd.last_hidden_state[torch.arange(batch_size), smi_attn_mask.sum(dim=1) - 1, :]

        pred = self.pred_nn(h).squeeze(1)

        return pred
    
class LlamaForMultiPropPred(nn.Module):
    def __init__(self, model_path, n_props=3, hf_path="ChemFM/ChemFM-1B", embed_dim=2048):
        super().__init__()

        self.model = LlamaModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_path)

        self.tokenizer.add_special_tokens({
            'pad_token': '<pad>'
        })
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.pred_nn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, n_props)
        )

        print(f"Loaded from {model_path}")

    def forward(self, smi_input_ids, smi_attn_mask):
        batch_size = smi_input_ids.shape[0]
        
        fwd = self.model(input_ids=smi_input_ids, attention_mask=smi_attn_mask)
        h = fwd.last_hidden_state[torch.arange(batch_size), smi_attn_mask.sum(dim=1) - 1, :]

        pred = self.pred_nn(h).squeeze(1)

        return pred
    
def train_for_prop_pred(model, train_dataloader, valid_dataloader, optimizer, epochs, ckpt_path, lora=False, save_full_model=False):

    model.to(device)
    best_valid_loss = float("inf")

    criterion = nn.L1Loss()
    
    for epoch in range(epochs):

        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            prop_pred = model(
                smi_input_ids=batch["smi_input_ids"],
                smi_attn_mask=batch["smi_attn_mask"]
            )

            loss = criterion(prop_pred, batch["prop"].squeeze(1))

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
                prop_pred = model(
                    smi_input_ids=batch["smi_input_ids"],
                    smi_attn_mask=batch["smi_attn_mask"]
                )

                loss = criterion(prop_pred, batch["prop"].squeeze(1))

                valid_loss += loss.item()
        mean_valid_loss = valid_loss / len(valid_dataloader)

        print(f"Epoch {epoch+1}/{epochs}: loss={mean_epoch_loss} | valid_loss={mean_valid_loss}")

        if mean_valid_loss < best_valid_loss:
            best_valid_loss = mean_valid_loss

            print(f"Saving at {ckpt_path}")
            if lora:
                if save_full_model:

                    merged_wrapper = LlamaForPropPred(
                        model_path=None,
                        lora=False,
                        embed_dim=model.embed_dim
                    )

                    merged_wrapper.model = model.model.merge_and_unload()
                    merged_wrapper.pred_nn.load_state_dict(model.pred_nn.state_dict())

                    torch.save(merged_wrapper.state_dict(), os.path.join(ckpt_path, "model.safetensors"))
                else:
                    merged = model.model.merge_and_unload()
                    merged.save_pretrained(ckpt_path)
            else:
                model.model.save_pretrained(ckpt_path)

def train_for_multi_prop_pred(model, train_dataloader, valid_dataloader, optimizer, epochs, ckpt_path):

    model.to(device)
    best_valid_loss = float("inf")
    
    for epoch in range(epochs):

        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            prop_pred = model(
                smi_input_ids=batch["smi_input_ids"],
                smi_attn_mask=batch["smi_attn_mask"]
            )

            loss = F.mse_loss(prop_pred, batch["prop"].squeeze(1))

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
                prop_pred = model(
                    smi_input_ids=batch["smi_input_ids"],
                    smi_attn_mask=batch["smi_attn_mask"]
                )

                loss = F.mse_loss(prop_pred, batch["prop"].squeeze(1))

                valid_loss += loss.item()
        mean_valid_loss = valid_loss / len(valid_dataloader)

        print(f"Epoch {epoch+1}/{epochs}: loss={mean_epoch_loss} | valid_loss={mean_valid_loss}")

        if mean_valid_loss < best_valid_loss:
            best_valid_loss = mean_valid_loss

            print(f"Saving at {ckpt_path}")
            model.model.save_pretrained(ckpt_path)