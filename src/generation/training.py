from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

import os

def train_causal_lm(
    model,
    tokenizer,
    train_dataset,
    valid_dataset,
    ckpt_steps,
    epochs,
    batch_size,
    output_dir="./output",
    lora=True
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

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator
    )

    trainer.train()

    if lora:
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(os.path.join(output_dir, "final"))