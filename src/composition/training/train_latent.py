import logging
import pathlib
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup

from src.composition.training.config.config import load_config
from src.composition.model.conductor_model import Conductor
from src.composition.training.training_data.dataloader.utils import (
    collate_fn,
    prepare_batch_for_model,
)
import torch
from torch.utils.data import DataLoader, ConcatDataset
from src.composition.training.training_data.dataloader.midi_dataset_loader import (
    MIDIDatasetLoader,
)

from src.composition.training.losses.instrument_activation_loss import (
    generate_instrument_activation_targets,
    instrument_activation_loss,
)

from src.composition.training.losses.instrument_counts_loss import (
    generate_instrument_counts_targets,
    instrument_counts_loss,
    generate_instance_mask_from_logits,
    generate_instance_mask_from_ground_truth,
)

from src.composition.training.losses.tempo_loss import (
    generate_tempo_targets,
    tempo_loss,
)


from bazel_tools.tools.python.runfiles import runfiles

from src.composition.training.dataset_resolver import get_dataset_files

import click
from tqdm import tqdm

from src.composition.game_genres import GameGenres
from src.composition.game_moods import GameMoods

logger = logging.getLogger(__name__)


def kl_anneal_weight(step, beta_max, warmup_steps, total_steps):
    if step < warmup_steps:
        return 0.0
    elif step < total_steps:
        return beta_max * (step - warmup_steps) / (total_steps - warmup_steps)
    else:
        return beta_max


@click.command()
@click.option(
    "--use-supervised", is_flag=True, help="Use supervised dataset for training"
)
@click.option(
    "--use-unsupervised", is_flag=True, help="Use unsupervised dataset for training"
)
@click.option(
    "--output-dir",
    required=True,
    type=pathlib.Path,
    help="Directory to store model checkpoints",
)
@click.option(
    "--dataset-cache-dir",
    type=pathlib.Path,
    help="Directory to cache dataset files",
)
def cli(
    use_supervised,
    use_unsupervised,
    output_dir,
    dataset_cache_dir=None,
):
    if use_supervised and use_unsupervised:
        raise ValueError(
            "Cannot use both supervised and unsupervised datasets at the same time."
        )

    if not use_supervised and not use_unsupervised:
        raise ValueError(
            "You must specify either --use-supervised or --use-unsupervised."
        )

    r = runfiles.Create()

    training_config = load_config(
        "_main/src/composition/training/config/latent_pretrain.yaml", r
    )

    datasets = []
    for dataset_name in training_config["datasets"]:
        dataset_files = get_dataset_files(dataset_name, r)
        cache_dir = dataset_cache_dir / dataset_name if dataset_cache_dir else None
        dataset = MIDIDatasetLoader(training_config, dataset_files, cache_dir)
        datasets.append(dataset)

    dataset = ConcatDataset(datasets)

    dataloader = DataLoader(
        dataset,
        batch_size=training_config["dataloader"]["batch_size"],
        shuffle=training_config["dataloader"]["shuffle"],
        num_workers=training_config["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    logger.info(f"Loaded datasets of size: {len(dataset)}")

    device = training_config["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your setup.")

    # Initialize model
    model = Conductor(
        training_config["model"]["latent_dim"],
        training_config["model"]["control_embed_dim"],
        len(GameGenres),
        len(GameMoods),
        training_config["num_instruments"],
        training_config["max_instrument_instances"],
    )
    model.to(device)

    optimizer_name = training_config["optimizer"]["name"]
    learning_rate = training_config["optimizer"]["lr"]
    weight_decay = training_config["optimizer"]["weight_decay"]

    optimizer_class = getattr(torch.optim, optimizer_name)

    # Initialize optimizer
    optimizer = optimizer_class(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Define training steps
    total_steps = training_config["training"]["epochs"] * len(dataloader)
    warmup_steps = int(0.1 * total_steps)  # 10% warmup

    # Cosine scheduler with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    writer = SummaryWriter(log_dir=output_dir / "tensorboard")

    # Training loop
    # TODO: Add more losses when supervised dataset is used
    model.train()
    for epoch in range(training_config["training"]["epochs"]):
        total_loss = 0.0
        for i, batch in tqdm(
            enumerate(dataloader),
            desc=f"Epoch {epoch + 1}/{training_config["training"]["epochs"]}",
        ):
            optimizer.zero_grad(set_to_none=True)

            inputs_and_control = prepare_batch_for_model(batch)
            inputs = inputs_and_control["input"]

            inputs = {k: v.to(device) for k, v in inputs.items()}

            if use_unsupervised:
                outputs = model(inputs)
            else:
                control_tokens = inputs_and_control["control"]
                assert (
                    control_tokens is not None
                ), "Control tokens must be provided for supervised training."
                control_tokens = {k: v.to(device) for k, v in control_tokens.items()}
                outputs = model(inputs, control_tokens)

            pred_tempos = outputs["tempos"]
            pred_instrument_counts_logits = outputs["instrument_counts_logits"]
            pred_instrument_activation_logits = outputs["instrument_activation_logits"]
            mu = outputs["latent_mu"]
            logvar = outputs["latent_logvar"]

            with torch.no_grad():
                # Generate targets
                target_tempos = generate_tempo_targets(inputs["bar_tempos"]).to(device)

                target_instrument_counts = generate_instrument_counts_targets(
                    inputs["track_mask"],
                    inputs["program_ids"],
                    training_config["num_instruments"],
                ).to(device)

                target_instrument_activation = generate_instrument_activation_targets(
                    inputs["bar_activations"],
                    inputs["track_mask"],
                    inputs["program_ids"],
                    training_config["num_instruments"],
                    training_config["max_instrument_instances"],
                ).to(device)

                instance_mask = (
                    generate_instance_mask_from_ground_truth(
                        inputs["track_mask"],
                        inputs["program_ids"],
                        training_config["num_instruments"],
                        training_config["max_instrument_instances"],
                    ).to(device)
                    if training_config["loss"][
                        "independent_instrument_activation_supervision"
                    ]
                    else generate_instance_mask_from_logits(
                        pred_instrument_counts_logits.detach(),
                    ).to(device)
                )

                # Unsqueeze mask over the bars
                instance_mask_exp = instance_mask.unsqueeze(1)

                # Unsqueeze attention over the instrumetns and instances
                attention_mask_exp = (
                    inputs["global_attention_mask"].unsqueeze(-1).unsqueeze(-1)
                ).to(device)

            # Compute losses
            tempo_loss_array = tempo_loss(pred_tempos, target_tempos)
            tempo_loss_value = tempo_loss_array[inputs["global_attention_mask"]].mean()

            instrument_counts_loss_value = instrument_counts_loss(
                pred_instrument_counts_logits, target_instrument_counts
            ).mean()

            # Reshape pred_instrument_activation_logits to match target_instrument_activation
            instrument_activation_loss_value = instrument_activation_loss(
                pred_instrument_activation_logits.view(
                    target_instrument_activation.shape
                ),
                target_instrument_activation,
                instance_mask_exp & attention_mask_exp,
            )

            latent_loss = (
                -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            ).mean()

            step_idx = epoch * len(dataloader) + i

            beta = kl_anneal_weight(
                step=step_idx,
                beta_max=training_config["loss"]["kl_max_weight"],
                warmup_steps=warmup_steps,
                total_steps=total_steps,
            )

            # Combine losses
            loss = (
                tempo_loss_value
                + instrument_counts_loss_value
                + instrument_activation_loss_value
                + beta * latent_loss
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            writer.add_scalar("Tempo Loss", tempo_loss_value.item(), step_idx)

            writer.add_scalar(
                "Instrument Counts Loss", instrument_counts_loss_value.item(), step_idx
            )

            writer.add_scalar(
                "Instrument Activation Loss",
                instrument_activation_loss_value.item(),
                step_idx,
            )

            writer.add_scalar("KL divergence Loss", latent_loss.item(), step_idx)
            writer.add_scalar("Total Loss", loss.item(), step_idx)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            },
            output_dir / f"checkpoint_{epoch}.pth",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    cli()
