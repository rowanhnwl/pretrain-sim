from src.generation.causal import *
from src.generation.data import *
from src.generation.training import *

from src.utils import split_train_valid

def main():
    model_path = "ChemFM/ChemFM-3B"
    n = 250000
    valid_ratio = 0.15
    ckpt_steps = 5000
    epochs = 3
    batch_size = 16

    model, tokenizer = load_causal_lm_and_tokenizer(
        model_path=model_path,
        hf_path=model_path,
        train_lora=True
    )

    dataset = ZincSMILESDataset(
        n=n,
        path="data/training/train_data.lmdb",
        tokenizer=tokenizer
    )

    train_dataset, valid_dataset = split_train_valid(
        dataset=dataset,
        valid_ratio=valid_ratio
    )

    train_causal_lm(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        ckpt_steps=ckpt_steps,
        epochs=epochs,
        batch_size=batch_size,
        lora=True
    )

if __name__ == "__main__":
    main()