from transformers import AutoTokenizer, LlamaForCausalLM, LlamaModel
import torch

from peft import get_peft_model, LoraConfig, TaskType, PeftModel

import os
import json

device = ("cuda" if torch.cuda.is_available() else "cpu")

def load_causal_lm_and_tokenizer(model_path, hf_path="ChemFM/ChemFM-1B", train_lora=False):
    tokenizer = AutoTokenizer.from_pretrained(hf_path)

    tokenizer.add_special_tokens({
        'pad_token': '<pad>'
    })

    if train_lora:
        lora_config = LoraConfig(
            r=4,
            lora_alpha=32,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        model = LlamaForCausalLM.from_pretrained(model_path)
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

        model = get_peft_model(model, lora_config)

        # Still need to train the full lm head
        for name, param in model.named_parameters():
            if "lm_head" in name:
                param.requires_grad = True

        model.print_trainable_parameters()
    else:
        model = LlamaForCausalLM.from_pretrained(model_path)

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    return model.to(device), tokenizer

def isolate_lm_grad(model: LlamaForCausalLM):
    for param in model.model.parameters():
        param.requires_grad = False

    for param in model.lm_head.parameters():
        param.requires_grad = True

def generate_smiles(input, n_samples, model, tokenizer, max_len=256):
    smiles_tokenized = tokenizer(input, return_tensors="pt").to(model.device)
    del smiles_tokenized["token_type_ids"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=smiles_tokenized["input_ids"],
            attention_mask=smiles_tokenized["attention_mask"],
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            num_return_sequences=n_samples,
            eos_token_id=tokenizer.eos_token_id,
            max_length=max_len,
            use_cache=False
        )

    outputs_list = []

    for output in outputs:
        output_smiles = tokenizer.decode(output, skip_special_tokens=True)
        output_smiles = output_smiles.replace(" ", "")

        if len(output_smiles) > 0:
            outputs_list.append(output_smiles)

    return outputs_list
