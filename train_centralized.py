import os
import warnings
from trl import KTOConfig, KTOTrainer
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import get_last_checkpoint

from utils.data import load_centralized_dataset
from utils.models import get_model, get_tokenizer_and_data_collator
from utils.training import set_seed
from utils.utils import parse_args_with_config, print_config, generate_deterministic_run_name

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class CompletionCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        # Create a file to indicate that training is complete
        open(os.path.join(args.output_dir, "training_complete.txt"), "a").close()


def main():
    """
    Main function to run a centralized training experiment.

    This function orchestrates the entire process:
    1. Parses command-line arguments to get the configuration file and any overrides.
    2. Sets the random seed for reproducibility.
    3. Loads the specified dataset and tokenizer.
    4. Generates a deterministic run name and save path based on the config.
    5. Checks if the training is already complete or can be resumed.
    6. Initializes the model using the configuration.
    7. Calculates weights for KTO loss.
    8. Configures and starts the Hugging Face Trainer.
    9. Saves the final model upon completion.
    """
    cfg, original_cfg = parse_args_with_config()
    print("Configuration:")
    print_config(cfg)

    set_seed(cfg.seed)
    print(f"Using seed: {cfg.seed}")

    # Determine run name and save path
    run_name = generate_deterministic_run_name(cfg, original_cfg)
    save_path = f"./models/centralized/{run_name}"

    # Check if training is already complete
    if os.path.exists(os.path.join(save_path, "training_complete.txt")):
        print(f"Training already complete for this configuration. Skipping run. Path: {save_path}")
        return

    # Check for last checkpoint
    last_checkpoint = get_last_checkpoint(save_path)
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")
        resume_from_checkpoint = last_checkpoint
    else:
        resume_from_checkpoint = None
        os.makedirs(save_path, exist_ok=True)


    # Load dataset and tokenizer
    dataset_path = cfg.dataset.path.format(cfg.dataset.name)
    dataset = load_centralized_dataset(dataset_path, cfg.get("dataset_index", None))
    tokenizer, data_collator = get_tokenizer_and_data_collator(
        cfg.model.name,
        cfg.model.use_fast_tokenizer,
        cfg.train.padding_side,
    )

    model = get_model(cfg.model)

    desirable_weight, undesirable_weight = 1.0, 1.0
    if dataset:
        num_desirable = sum(dataset["label"]) * 3
        num_undesirable = (dataset.num_rows - num_desirable / 3) * 4
        if num_desirable > 0 and num_undesirable > 0:
            if num_desirable < num_undesirable:
                desirable_weight = num_undesirable / num_desirable
            else:
                undesirable_weight = num_desirable / num_undesirable
    print(f"Desirable weight: {desirable_weight:.2f}, Undesirable weight: {undesirable_weight:.2f}")

    training_args = KTOConfig(
        **cfg.train.training_arguments,
        output_dir=save_path,
        desirable_weight=desirable_weight,
        undesirable_weight=undesirable_weight,
        seed=cfg.seed,
    )

    trainer = KTOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[CompletionCallback()],
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
