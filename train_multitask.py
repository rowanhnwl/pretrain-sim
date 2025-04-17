from src.multitask.data import *
from src.multitask.models import LlamaMultitask
from src.multitask.training import *

from src.utils import *

def main():
    n = 250000
    valid_ratio = 0.15
    ckpt_steps = 5000
    epochs = 1
    batch_size = 8
    lora=True

    task_weights = (0.1, 0.9)

    base_model, tokenizer = load_llama_and_tokenizer_for_training(
        model_path="checkpoints/tpsa_3b"
    )

    dataset = ZincSMILESWithTPSA(n=n, tokenizer=tokenizer)
    train_dataset, valid_dataset = split_train_valid(dataset=dataset, valid_ratio=valid_ratio)

    model = LlamaMultitask(
        config=base_model.config,
        base_model=base_model,
        task_weights=task_weights,
        lora=lora
    )

    train_multitask(
        model,
        tokenizer,
        train_dataset,
        valid_dataset,
        ckpt_steps,
        epochs,
        batch_size,
        lora=lora
    )

if __name__ == "__main__":
    main()