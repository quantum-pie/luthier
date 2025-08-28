import logging
import pathlib
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup

from src.composition.training.config.config import load_config
from src.composition.model.conductor_model import Conductor
from src.composition.model.input_embeddings import InputEmbeddings
from src.composition.training.training_data.dataloader.utils import (
    collate_fn,
    prepare_batch_for_model,
)
import torch
from torch.utils.data import DataLoader, ConcatDataset
from src.composition.training.training_data.dataloader.midi_dataset_loader import (
    MIDIDatasetLoader,
)

from src.composition.training.losses.instrument_density_loss import (
    generate_instrument_density_targets,
    instrument_density_loss,
)

from src.composition.training.losses.instrument_counts_loss import (
    generate_instrument_counts_targets,
    instrument_counts_loss,
)

from src.composition.training.losses.tempo_loss import (
    generate_tempo_targets,
    tempo_loss,
)


from bazel_tools.tools.python.runfiles import runfiles

from src.composition.training.dataset_resolver import get_dataset_files

import click
from tqdm import tqdm

import numpy as np

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


def kl_loss(mu_posterior, logvar_posterior, mu_prior, logvar_prior):
    """
    KL(q||p) where
      q = N(mu_posterior, diag(exp(logvar_posterior)))
      p = N(mu_prior,     diag(exp(logvar_prior)))

    Args:
        mu_posterior:   [B, D]
        logvar_posterior: [B, D]
        mu_prior:       [B, D]
        logvar_prior:   [B, D]
        reduction: "mean" | "sum" | "none"
    Returns:
        Scalar loss if reduction != "none", else per-sample KL [B]
    """
    # variances
    var_post = torch.exp(logvar_posterior)
    var_prior = torch.exp(logvar_prior)

    # KL(q||p) per-dimension
    # 0.5 * [ log(|Σ_p|/|Σ_q|) - D + tr(Σ_p^{-1} Σ_q) + (μ_p - μ_q)^T Σ_p^{-1} (μ_p - μ_q) ]
    log_det_ratio = logvar_prior - logvar_posterior  # [B, D]
    trace_term = var_post / var_prior  # [B, D]
    mean_diff_sq = (mu_prior - mu_posterior).pow(2) / var_prior

    kl_per_dim = log_det_ratio + trace_term + mean_diff_sq - 1.0
    kl_per_sample = 0.5 * kl_per_dim.sum(dim=-1)  # [B]

    return kl_per_sample


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

    device = training_config["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your setup.")

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
        pin_memory=False,
    )

    logger.info(f"Loaded datasets of size: {len(dataset)}")

    # Initialize model
    conductor_model = Conductor(
        training_config["model"]["latent_dim"],
        training_config["model"]["control_embed_dim"],
        training_config["model"]["hidden_dim"],
        len(GameGenres),
        len(GameMoods),
        training_config["num_instruments"],
        training_config["max_instrument_instances"],
    )
    conductor_model.to(device)

    input_embeddings_model = InputEmbeddings(
        training_config["model"]["hidden_dim"],
        training_config["vocab"]["pitch_vocab_size"],
        training_config["vocab"]["velocity_vocab_size"],
        training_config["num_instruments"],
    )
    input_embeddings_model.to(device)

    optimizer_name = training_config["optimizer"]["name"]
    learning_rate = training_config["optimizer"]["lr"]
    weight_decay = training_config["optimizer"]["weight_decay"]

    optimizer_class = getattr(torch.optim, optimizer_name)

    # Initialize optimizer
    optimizer = optimizer_class(
        list(conductor_model.parameters()) + list(input_embeddings_model.parameters()),
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
    conductor_model.train()
    input_embeddings_model.train()

    try:
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

                input_embeddings = input_embeddings_model(inputs)
                if use_unsupervised:
                    outputs = conductor_model(inputs, input_embeddings)
                else:
                    control_tokens = inputs_and_control["control"]
                    assert (
                        control_tokens is not None
                    ), "Control tokens must be provided for supervised training."
                    control_tokens = {
                        k: v.to(device) for k, v in control_tokens.items()
                    }
                    outputs = conductor_model(inputs, input_embeddings, control_tokens)

                pred_tempos = outputs["tempos"]
                pred_instrument_counts_rates = outputs["instrument_counts_rates"]
                pred_instrument_density_logits = outputs["instrument_density_logits"]

                mu_prior = outputs["mu_prior"]
                logvar_prior = outputs["logvar_prior"]
                mu_posterior = outputs["mu_posterior"]
                logvar_posterior = outputs["logvar_posterior"]

                with torch.no_grad():
                    # Generate targets
                    target_tempos = generate_tempo_targets(inputs["bar_tempos"]).to(
                        device
                    )

                    target_instrument_counts = generate_instrument_counts_targets(
                        inputs["track_mask"],
                        inputs["program_ids"],
                        training_config["num_instruments"],
                    ).to(device)

                    target_instrument_density = generate_instrument_density_targets(
                        inputs["bar_activations"],
                        inputs["track_mask"],
                        inputs["program_ids"],
                        target_instrument_counts,
                        training_config["num_instruments"],
                    ).to(device)

                # Compute losses
                tempo_loss_array = tempo_loss(pred_tempos, target_tempos)
                tempo_loss_value = tempo_loss_array[
                    inputs["global_attention_mask"]
                ].mean()

                instrument_counts_loss_value = instrument_counts_loss(
                    pred_instrument_counts_rates, target_instrument_counts
                ).mean()

                full_mask = inputs["global_attention_mask"].unsqueeze(-1) & (
                    target_instrument_counts > 0
                ).unsqueeze(
                    1
                )  # [B, T, P]

                instrument_density_loss_value = instrument_density_loss(
                    pred_instrument_density_logits,
                    target_instrument_density,
                    target_instrument_counts,
                )[full_mask].mean()

                latent_loss = kl_loss(
                    mu_posterior, logvar_posterior, mu_prior, logvar_prior
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
                    + instrument_density_loss_value
                    + beta * latent_loss
                )

                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

                writer.add_scalar("Tempo Loss", tempo_loss_value.item(), step_idx)

                writer.add_scalar(
                    "Instrument Counts Loss",
                    instrument_counts_loss_value.item(),
                    step_idx,
                )

                writer.add_scalar(
                    "Instrument Density Loss",
                    instrument_density_loss_value.item(),
                    step_idx,
                )

                writer.add_scalar("KL divergence Loss", latent_loss.item(), step_idx)
                writer.add_scalar("Total Loss", loss.item(), step_idx)

                logger.info(
                    f"Batch {i + 1}/{len(dataloader)} processed. Total loss: {total_loss / (i + 1)}"
                )

            torch.save(
                {
                    "conductor_model_state_dict": conductor_model.state_dict(),
                    "input_embeddings_model_state_dict": input_embeddings_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                    "config": training_config,
                },
                output_dir / f"checkpoint_{epoch}.pth",
            )
    except KeyboardInterrupt:
        logger.info("Training interrupted. Saving the last checkpoint.")

    torch.save(
        {
            "conductor_model_state_dict": conductor_model.state_dict(),
            "input_embeddings_model_state_dict": input_embeddings_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
            "config": training_config,
        },
        output_dir / "model.pth",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
