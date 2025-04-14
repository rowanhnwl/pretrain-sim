from transformers import LlamaModel, LlamaForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch

from src.multitask.data import MultiTaskCollator

import os

device = ("cuda" if torch.cuda.is_available() else "cpu")

class MultitaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]

        if self.state.global_step % self.args.logging_steps == 0 and self.state.global_step > 0:
            self.log({
                "lm_loss": outputs["lm_loss"].item(),
                "prop_loss": outputs["prop_loss"].item()
            })

        return (loss, outputs) if return_outputs else loss

def load_llama_and_tokenizer_for_training(
    model_path,
    hf_path="ChemFM/ChemFM-1B"
):
    tokenizer = AutoTokenizer.from_pretrained(hf_path)

    tokenizer.add_special_tokens({
        'pad_token': '<pad>'
    })

    model = LlamaModel.from_pretrained(model_path)

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    return model.to(device), tokenizer

def save_as_causal_lm(model, output_dir, lora):
    base_model = model.llama
    lm_head = model.lm_head

    if lora:
        base_model = base_model.merge_and_unload()

    causal_lm = LlamaForCausalLM(model.config)

    causal_lm.model.load_state_dict(base_model.state_dict())
    causal_lm.lm_head.load_state_dict(lm_head.state_dict())

    causal_lm.save_pretrained(os.path.join(output_dir, "causal"))

def train_multitask(
    model,
    tokenizer,
    train_dataset,
    valid_dataset,
    ckpt_steps,
    epochs,
    batch_size,
    output_dir="./output",
    lora=False
):
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=ckpt_steps,
        save_steps=ckpt_steps,
        fp16=True
    )

    collator = MultiTaskCollator(tokenizer)

    trainer = MultitaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator
    )

    trainer.train()

    save_as_causal_lm(model, output_dir, lora)