from pathlib import Path
from src.composition.midi.velocity_quantizer import VelocityQuantizer
from src.composition.training.training_data.dataloader.base_loader import DatasetLoader
from src.composition.midi.tokenizer import MultiTrackMidiTokenizer
import numpy as np


class UnsupervisedMIDILoader(DatasetLoader):
    def __init__(self, config, files):
        self.config = config
        self.filepaths = sorted(files)
        velocity_quantizer = VelocityQuantizer(
            velocity_bins=config["vocab"]["velocity_vocab_size"]
        )
        self.tokenizer = MultiTrackMidiTokenizer(velocity_quantizer=velocity_quantizer, max_instrument_instances=config["max_instrument_instances"])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        tokenizer_result = self.tokenizer.encode(self.filepaths[idx])
        while tokenizer_result is None:
            random_idx = np.random.randint(0, len(self))
            tokenizer_result = self.tokenizer.encode(self.filepaths[random_idx])

        return tokenizer_result
