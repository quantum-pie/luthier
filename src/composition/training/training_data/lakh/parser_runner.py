import click
import logging
from src.composition.training.training_data.lakh.parser import LakhParser
from pathlib import Path
import os

@click.command()
@click.option("--dataset_dir", type=Path, required=True)
def cli(dataset_dir):
    LakhParser(dataset_dir)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
