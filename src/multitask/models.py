from transformers import PreTrainedModel, LlamaConfig, ModelOutput, LlamaModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Union

import torch.nn.functional as F
import torch.nn as nn

class LlamaForMultiTask(PreTrainedModel):
    config_class = LlamaConfig

    def __init__(self, config, num_properties, task_weights=(1.0, 1.0)):
        super().__init__(config)

        self.llama_model = LlamaModel(config)
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.prop_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, num_properties)
        )

        self.alpha, self.beta = task_weights

    def forward(self, input_ids, attn_mask=None, labels=None, props=None, task=None):
        h = self.llama_model(
            input_ids=input_ids,
            attention_mask=attn_mask
        )

        lhs = h.last_hidden_state[:, -1, :]
        loss = None

        if task == "causal":
            logits = self.lm_head(lhs)
            if labels is not None:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss_fn = nn.CrossEntropyLoss()
                causal_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                loss = self.alpha * causal_loss

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=None,
                hidden_states=h.hidden_states,
                attentions=h.attentions
            )
        elif task == "regression":
            prop_preds = self.prop_head(lhs)
            
            loss_fn = nn.MSELoss()
            prop_loss = loss_fn(prop_preds.view(-1), props.view(-1))

            loss = self.beta * prop_loss

            return ModelOutput(loss=loss, prop_preds=prop_preds)