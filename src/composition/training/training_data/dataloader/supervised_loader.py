import json
from src.composition.midi.tokenizer import MultiTrackMidiTokenizer
from src.composition.training.training_data.dataloader.base_loader import DatasetLoader
from src.composition.midi.velocity_quantizer import VelocityQuantizer


# TODO: Unfinished implementation of SupervisedMIDILoader
class SupervisedMIDILoader(DatasetLoader):
    def __init__(self, config, files):
        self.config = config
        self.filepaths = sorted(files)
        velocity_quantizer = VelocityQuantizer(
            velocity_bins=config["vocab"]["velocity_vocab_size"]
        )
        self.tokenizer = MultiTrackMidiTokenizer(velocity_quantizer=velocity_quantizer)

    def __len__(self):
        return len(self.filepaths)

    def _load_tags(self, tag_file):
        if tag_file.endswith(".json"):
            return json.load(open(tag_file))
        elif tag_file.endswith(".csv"):
            # Optional: implement CSV tag reader
            raise NotImplementedError
        else:
            raise ValueError("Unsupported tag file format")

    def __getitem__(self, idx):
        tokenizer_result = self.tokenizer.encode(self.filepaths[idx])
        while tokenizer_result is None:
            random_idx = np.randint(0, len(self))
            tokenizer_result = self.tokenizer.encode(self.filepaths[random_idx])
        return tokenizer_result
