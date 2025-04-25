from transformers import PreTrainedModel, LlamaConfig
import torch.nn as nn
import torch

from peft import get_peft_model, LoraConfig

class LlamaMultitask(PreTrainedModel):
    config_class = LlamaConfig

    def __init__(self, config, base_model, task_weights=(1.0, 1.0), lora=False):
        super().__init__(config)

        self.alpha, self.beta = task_weights

        if lora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                bias="none"
            )

            self.model = get_peft_model(base_model, lora_config)
        else:
            self.model = base_model
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.property_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1)
        )

    def forward(self, input_ids, attn_mask=None, labels=None, tpsa=None):
        outputs = self.llama(input_ids=input_ids, attention_mask=attn_mask)
        hidden_states = outputs.last_hidden_state

        batch_size = input_ids.shape[0]

        lm_logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            lm_loss_fct = nn.CrossEntropyLoss()
            lm_loss = lm_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        prop_preds = self.property_head(hidden_states[torch.arange(batch_size), attn_mask.sum(dim=1) - 1, :])
        if tpsa is not None:
            prop_loss_fct = nn.L1Loss()
            prop_loss = prop_loss_fct(prop_preds.view(-1), tpsa.view(-1))

        loss = self.alpha * lm_loss + self.beta * prop_loss

        return {
            "loss": loss,
            "logits": lm_logits,
            "lm_loss": lm_loss,
            "prop_loss": prop_loss
        }