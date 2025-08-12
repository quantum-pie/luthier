from pathlib import Path
import pickle
from src.composition.midi.velocity_quantizer import VelocityQuantizer
from src.composition.training.training_data.dataloader.base_loader import DatasetLoader
from src.composition.midi.tokenizer import MultiTrackMidiTokenizer
from src.composition.training.training_data.dataloader.utils import atomic_write
import numpy as np


class MIDIDatasetLoader(DatasetLoader):
    def __init__(self, config, files, cache_dir):
        self.config = config
        self.filepaths = sorted(files)
        velocity_quantizer = VelocityQuantizer(
            velocity_bins=config["vocab"]["velocity_vocab_size"]
        )
        self.tokenizer = MultiTrackMidiTokenizer(
            velocity_quantizer=velocity_quantizer,
            max_instrument_instances=config["max_instrument_instances"],
        )

        self.cache_dir = cache_dir
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        if self.cache_dir:
            cache_file = self.cache_dir / f"{idx}.pkl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

        tokenizer_result = self.tokenizer.encode(self.filepaths[idx])
        while tokenizer_result is None:
            random_idx = np.random.randint(0, len(self))
            tokenizer_result = self.tokenizer.encode(self.filepaths[random_idx])

        result = {
            "input": tokenizer_result,
            "control": None,  # TODO: Implement control tokens from metadata
        }

        if self.cache_dir:
            cache_file = self.cache_dir / f"{idx}.pkl"
            write_fn = lambda f: pickle.dump(result, f)
            atomic_write(cache_file, write_fn)

        return result
