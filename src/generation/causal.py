from transformers import AutoTokenizer, LlamaForCausalLM

import torch

device = ("cuda" if torch.cuda.is_available() else "cpu")

def load_causal_lm_and_tokenizer(model_path, hf_path="ChemFM/ChemFM-1B"):
    tokenizer = AutoTokenizer.from_pretrained(hf_path)

    tokenizer.add_special_tokens({
        'pad_token': '<pad>'
    })
    
    model = LlamaForCausalLM.from_pretrained(model_path)

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    return model.to(device), tokenizer

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
            max_length=max_len
        )

    outputs_list = []

    for output in outputs:
        output_smiles = tokenizer.decode(output, skip_special_tokens=True)
        output_smiles = output_smiles.replace(" ", "")

        if len(output_smiles) > 0:
            outputs_list.append(output_smiles)

    return outputs_list
