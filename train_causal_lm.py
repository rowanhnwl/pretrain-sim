from src.generation.causal import *
from src.generation.data import *
from src.generation.training import *

def main():
    model_path = "checkpoints/tpsa"
    data_path = "data/training/train_data.lmdb"
    n = 100000
    valid_ratio = 0.15
    ckpt_steps = 10000
    epochs = 5
    batch_size = 16

    model, tokenizer = load_lora_causal_lm_and_tokenizer(
        model_path=model_path
    )

    dataset = SMILESDataset(
        filepath=data_path,
        n=n
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
        batch_size=batch_size
    )

if __name__ == "__main__":
    main()