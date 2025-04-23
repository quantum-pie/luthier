import click
import logging
from src.composition.training.training_data.lakh.loader import LakhLoader
from pathlib import Path

@click.command()
@click.option("--dataset_dir", type=Path, required=True)
def cli(dataset_dir):
    LakhLoader(dataset_dir)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
