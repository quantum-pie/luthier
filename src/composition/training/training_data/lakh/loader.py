
from pathlib import Path
import json
import os
import logging

from dataclasses import dataclass
from enum import Enum

from src.composition.training.training_data.lakh.hdf5_getters import open_h5_file_read, get_song_hotttnesss

logger = logging.getLogger(__name__)

class MatchedMidiType(Enum):
    matched = 0,
    aligned = 1

@dataclass
class LakhMetadata:
    """
    This class represents the metadata for a Lakh MIDI file.
    """
    md5: str
    filename: str
    match_score: float
    # Add other relevant fields as needed

class LakhLoader:
    """
    This class is responsible for loading the Lakh MIDI dataset.
    """

    def __init__(self, data_dir: Path):
        self._matched_data_dir = data_dir / "lmd_matched"
        self._aligned_data_dir = data_dir / "lmd_aligned"
        self._metadata_dir = data_dir / "lmd_matched_h5"

        with open(data_dir / "md5_to_paths.json", "r") as f:
            logger.info("Loading MD5 to filename map...")
            self._md5_to_filename = json.load(f)

        with open(data_dir / "match_scores.json", "r") as f:
            logger.info("Loading matching scores...")
            self._match_scores = json.load(f)

        self.build_index()

    @staticmethod
    def song_id_to_dir(song_id: str) -> str:
        return os.path.join(song_id[2], song_id[3], song_id[4], song_id)
    
    def song_id_to_metada_path(self, song_id: str):
        return self._metadata_dir / Path(LakhLoader.song_id_to_dir(song_id)).with_suffix('.h5') 

    def song_id_to_midi_path(self, song_id: str, midi_md5: str, type: MatchedMidiType):
        base_dir = self._matched_data_dir if type == MatchedMidiType.matched else self._aligned_data_dir
        return base_dir / Path(LakhLoader.song_id_to_dir(song_id)) / Path(midi_md5).with_suffix('.mid')
        
    def build_index(self):
        """
        Build an index of metadata for the Lakh MIDI dataset.
        """
        self._index = {}
        for song_id, matches in self._match_scores.items():
            for md5 in matches.keys():
                original_filename = self._md5_to_filename[md5]
                location = self.song_id_to_midi_path(song_id, md5, MatchedMidiType.aligned)
                metadata_path = self.song_id_to_metada_path(song_id)

                meta = open_h5_file_read(metadata_path)
                logger.info(f"Song hotttness: {get_song_hotttnesss(meta)}")