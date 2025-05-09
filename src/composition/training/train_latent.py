import logging
import pathlib

from src.composition.training.config.config import load_config
from src.composition.model.conductor_model import Conductor
from src.composition.training.training_data.dataloader.utils import (
    collate_fn,
    prepare_batch_for_model,
)
import torch
from torch.utils.data import DataLoader, ConcatDataset
from src.composition.training.training_data.dataloader.supervised_loader import (
    SupervisedMIDILoader,
)
from src.composition.training.training_data.dataloader.unsupervised_loader import (
    UnsupervisedMIDILoader,
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


@click.command()
@click.option(
    "--use-supervised", is_flag=True, help="Use supervised dataset for training"
)
@click.option(
    "--use-unsupervised", is_flag=True, help="Use unsupervised dataset for training"
)
def cli(
    use_supervised,
    use_unsupervised,
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
        dataset = (
            SupervisedMIDILoader(training_config, dataset_files)
            if use_supervised
            else UnsupervisedMIDILoader(training_config, dataset_files)
        )
        datasets.append(dataset)

    dataset = ConcatDataset(datasets)

    dataloader = DataLoader(
        dataset,
        batch_size=training_config["dataloader"]["batch_size"],
        shuffle=training_config["dataloader"]["shuffle"],
        num_workers=training_config["num_workers"],
        collate_fn=collate_fn,
    )

    print("Loaded datasets:", len(dataset))

    device = training_config["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your setup.")

    control_vocab_size = {
        "genre": len(GameGenres),
        "mood": len(GameMoods),
    }

    # Initialize model
    model = Conductor(
        training_config["model"]["latent_dim"],
        training_config["model"]["control_embed_dim"],
        control_vocab_size,
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

    # Training loop
    # TODO: Add more losses when supervised dataset is used
    model.train()
    for epoch in range(training_config["training"]["epochs"]):
        total_loss = 0.0
        for batch in tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{training_config["training"]["epochs"]}",
        ):
            print("Processing batch")
            inputs = prepare_batch_for_model(batch)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            optimizer.zero_grad()
            outputs = model(inputs)

            print("got outputs", outputs.keys())

            ############################## INVESTIGATE HERE ##############################
            # /home/quantum-pie/Projects/Luthier/datasets/lakh/lmd_full/c/cab976b8b9901f4d9bf49e1d86173a56.mid

            pred_tempos = outputs["tempos"]
            pred_instrument_counts_logits = outputs["instrument_counts_logits"]
            pred_instrument_activation_logits = outputs["instrument_activation_logits"]
            mu = outputs["latent_mu"]
            logvar = outputs["latent_logvar"]

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
                    pred_instrument_counts_logits,
                ).to(device)
            )

            # Unsqueeze mask over the bars
            instance_mask_exp = instance_mask.unsqueeze(1)

            # Unsqueeze attention over the instrumetns and instances
            attention_mask_exp = (
                inputs["global_attention_mask"].unsqueeze(-1).unsqueeze(-1)
            ).to(device)

            print("att mask device", attention_mask_exp.device)

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
                -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            ).mean()

            # Combine losses
            loss = (
                tempo_loss_value
                + instrument_counts_loss_value
                + instrument_activation_loss_value
                + latent_loss
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
