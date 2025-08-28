import logging
import pathlib

from src.composition.midi.tempo_normalizer import TempoNormalizer
from src.composition.model.conductor_model import Conductor

import torch

import click
import numpy as np

from src.composition.game_genres import GameGenres
from src.composition.game_moods import GameMoods

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model-path",
    required=True,
    type=pathlib.Path,
    help="Path to the conductor model checkpoint",
)
def cli(
    model_path: pathlib.Path,
):
    model_state = torch.load(model_path)

    config = model_state["config"]
    device = config["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your setup.")

    for _ in range(1):
        # Reinitialize model on each iteration. By design the latent state is not resampled
        # as long as the control input doesn't change. This preservers the same intent throughout the
        # piece. For demonstration purposes we reinitialize the model multiple times to see variation.
        conductor = Conductor(
            config["model"]["latent_dim"],
            config["model"]["control_embed_dim"],
            config["model"]["hidden_dim"],
            len(GameGenres),
            len(GameMoods),
            config["num_instruments"],
            config["max_instrument_instances"],
        )

        conductor.load_state_dict(model_state["conductor_model_state_dict"])
        conductor.to(device)
        conductor.eval()

        with torch.no_grad():
            output = conductor.forward_step(
                bar_position=torch.tensor(0, device=device, dtype=torch.float32)
            )
            instruments_count_rates = output["instrument_counts_rates"]
            instrument_counts = torch.poisson(instruments_count_rates)

            logger.info(
                f"Tempo (bpm): {TempoNormalizer().unnormalize_bpm(output["tempo"]).item()}"
            )

            for program_id, count in enumerate(instrument_counts):
                if count > 0:
                    logger.info(f"Program ID {program_id}: {count} instances")

        SEQ_LEN = 256
        time_signature = 4  # 4/4 time
        instrument_density_logits = torch.zeros(
            (config["num_instruments"], SEQ_LEN), device=device
        )
        for bar_position in range(1, SEQ_LEN):
            with torch.no_grad():
                bar_position_torch = torch.tensor(
                    bar_position * time_signature, device=device, dtype=torch.float32
                )
                output = conductor.forward_step(bar_position=bar_position_torch)
                for program_id, count in enumerate(instrument_counts):
                    if count == 0:
                        continue

                    logit = output["instrument_density_logits"][program_id]
                    instrument_density_logits[program_id, bar_position] = logit.item()

        for program_id, count in enumerate(instrument_counts):
            if count > 0:
                instrument_density_probs = torch.sigmoid(
                    instrument_density_logits[program_id]
                )
                instruments_density = (
                    torch.round(instrument_density_probs * count).cpu().numpy()
                )
                if all(instruments_density < int(count.item())):
                    logger.info(f"Program ID {program_id} was never activated enough.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
