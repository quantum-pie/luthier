from pathlib import Path
import json
import os
import logging
import yaml
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

import hdbscan
import umap.umap_ as umap

from enum import Enum

from src.composition.training.training_data.lakh.hdf5_getters import *
from bazel_tools.tools.python.runfiles import runfiles

logger = logging.getLogger(__name__)

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MODE_NAMES = ["Minor", "Major"]


class MatchedMidiType(Enum):
    matched = (0,)
    aligned = 1


class LakhParser:
    """
    This class is responsible for loading the Lakh MIDI dataset.
    """

    def __init__(self, data_dir: Path):
        r = runfiles.Create()
        config_path = Path(
            r.Rlocation("_main/src/composition/training/training_data/lakh/config.yaml")
        )

        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)["lakh_loader"]

        self._root = data_dir
        self._matched_data_dir = data_dir / "lmd_matched"
        self._aligned_data_dir = data_dir / "lmd_aligned"
        self._metadata_dir = data_dir / "lmd_matched_h5"
        self._cooccurrence_path = data_dir / "cooccurrence.pkl"
        self._cooccurrence_data = {}
        self._tag_clustering_path = data_dir / "clustering.pkl"
        self._tag_clustering = {}

        with open(data_dir / "md5_to_paths.json", "r") as f:
            logger.info("Loading MD5 to filename map...")
            self._md5_to_filename = json.load(f)

        with open(data_dir / "match_scores.json", "r") as f:
            logger.info("Loading matching scores...")
            self._match_scores = json.load(f)

        self.build_tag_cooccurrence()
        self.cluster_tags()
        self.classify_songs_into_tag_clusters()

    @staticmethod
    def song_id_to_dir(song_id: str) -> str:
        return os.path.join(song_id[2], song_id[3], song_id[4], song_id)

    def song_id_to_metada_path(self, song_id: str):
        return self._metadata_dir / Path(
            LakhParser.song_id_to_dir(song_id)
        ).with_suffix(".h5")

    def song_id_to_midi_path(self, song_id: str, midi_md5: str, type: MatchedMidiType):
        base_dir = (
            self._matched_data_dir
            if type == MatchedMidiType.matched
            else self._aligned_data_dir
        )
        return (
            base_dir
            / Path(LakhParser.song_id_to_dir(song_id))
            / Path(midi_md5).with_suffix(".mid")
        )

    def calculate_cluster_scores(self, artist_terms, weights):
        cluster_scores = {}
        for tag, weight in zip(artist_terms, weights):
            labels_list = self._tag_clustering["labels_list"]
            tag_index = self._cooccurrence_data["tag_index"]

            idx = tag_index[tag.decode("utf-8").lower()]
            cluster = labels_list[idx]
            if cluster == -1:
                continue

            if cluster in cluster_scores:
                cluster_scores[cluster] += weight
            else:
                cluster_scores[cluster] = weight

        return cluster_scores

    def cluster_tags(self):
        if Path.exists(self._tag_clustering_path):
            logger.info("Found cached clustering data!")
            with open(self._tag_clustering_path, "rb") as f:
                self._tag_clustering = pickle.load(f)
                return

        # Step 1: Filter out weakly-connected tags percentile
        co_matrix = self._cooccurrence_data["tag_cooccurrence"]
        connectivity = co_matrix.sum(axis=1) - np.diag(co_matrix)

        plt.hist(connectivity, bins=100, log=True)
        plt.xlabel("Total co-occurrences")
        plt.ylabel("Number of tags")
        plt.title("Tag Connectivity Histogram")
        plt.tight_layout()
        plt.savefig(self._root / "connectivity_distribution.png", dpi=300)
        plt.close()

        cutoff = np.percentile(connectivity, 10)
        valid_indices = np.where(connectivity >= cutoff)[0]

        filtered_cooccurrence = self._cooccurrence_data["tag_cooccurrence"][
            np.ix_(valid_indices, valid_indices)
        ]

        # Step 2: Normalize tag co-occurrence vectors
        X_norm = normalize(filtered_cooccurrence.astype(float), norm="l2", axis=1)

        # Step 3: Reduce dimensionaly
        reducer = umap.UMAP(
            n_neighbors=50,
            min_dist=0.5,
            n_components=10,
            metric="cosine",
            random_state=42,
        )
        X_umap = reducer.fit_transform(X_norm)

        # Step 4: Cluster
        db = hdbscan.HDBSCAN(min_cluster_size=self._config["min_cluster_size"])
        labels = np.array([-1] * len(self._cooccurrence_data["tag_counts"]))
        labels[valid_indices] = db.fit_predict(X_umap)

        labels_index = {}
        for idx, label in enumerate(labels):
            if label == -1:
                continue
            if label in labels_index:
                labels_index[label].append(idx)
            else:
                labels_index[label] = [idx]

        logger.info(f"Num unique clusters: {len(labels_index)}")

        labels_names = {}
        for label, tag_indices in labels_index.items():
            cluster_tags_counts = [
                self._cooccurrence_data["tag_counts"][i] for i in tag_indices
            ]
            most_frequent_tag_idx = np.argmax(cluster_tags_counts)
            cluster_name = self._cooccurrence_data["tag_list"][
                tag_indices[most_frequent_tag_idx]
            ]
            labels_names[label] = cluster_name
            logger.info(f"Cluster id {label} has name {cluster_name}")

        result = {
            "labels_list": labels,
            "labels_index": labels_index,
            "labels_names": labels_names,
        }
        self._tag_clustering = result
        with open(self._tag_clustering_path, "wb") as f:
            pickle.dump(result, f)

    def build_tag_cooccurrence(self):
        if Path.exists(self._cooccurrence_path):
            logger.info("Found cached co-occurrence data!")
            with open(self._cooccurrence_path, "rb") as f:
                self._cooccurrence_data = pickle.load(f)
                return

        tag_set = set()
        logger.info("Collecting tags...")
        for song_id, matches in tqdm(self._match_scores.items()):
            for _, score in matches.items():
                if score < self._config["min_match_score"]:
                    continue

                metadata_path = self.song_id_to_metada_path(song_id)
                with open_h5_file_read(metadata_path) as meta:
                    artist_terms = get_artist_terms(meta)
                    tag_set.update(tag.decode("utf-8").lower() for tag in artist_terms)

        tag_list = list(tag_set)
        tag_index = {tag: i for i, tag in enumerate(tag_list)}

        logger.info(f"Found {len(tag_index)} unique tags.")

        tag_counts = np.zeros(len(tag_list), np.uint32)
        tag_cooccurrence = np.zeros((len(tag_list), len(tag_list)), np.uint32)

        logger.info("Calculating co-occurrence...")
        for song_id, matches in tqdm(self._match_scores.items()):
            for _, score in matches.items():
                if score < self._config["min_match_score"]:
                    continue

                metadata_path = self.song_id_to_metada_path(song_id)
                with open_h5_file_read(metadata_path) as meta:
                    artist_terms = get_artist_terms(meta)
                    tags = [tag.decode("utf-8").lower() for tag in artist_terms]
                    indices = [tag_index[t] for t in tags]
                    for i in indices:
                        tag_counts[i] += 1
                        for j in indices:
                            tag_cooccurrence[i, j] += 1

        result = {
            "tag_list": tag_list,
            "tag_index": tag_index,
            "tag_counts": tag_counts,
            "tag_cooccurrence": tag_cooccurrence,
            "tag_cosine_similarity": cosine_similarity(tag_cooccurrence),
        }
        self._cooccurrence_data = result
        with open(self._cooccurrence_path, "wb") as f:
            pickle.dump(result, f)

    @staticmethod
    def cluster_entropy(distribution: dict) -> float:
        probs = np.array(list(distribution.values()))
        probs = probs[probs > 0]

        if len(probs) == 0:
            return 1.0

        if len(probs) == 1:
            return 0.0

        return -np.sum(probs * np.log(probs)) / np.log(len(distribution))

    def classify_songs_into_tag_clusters(self):
        songs_per_cluster = np.zeros(
            len(self._tag_clustering["labels_index"]), np.uint32
        )
        for song_id, matches in tqdm(self._match_scores.items()):
            for md5, score in matches.items():
                if score < self._config["min_match_score"]:
                    continue

                # original_filename = self._md5_to_filename[md5]
                # location = self.song_id_to_midi_path(
                #     song_id, md5, MatchedMidiType.aligned
                # )
                metadata_path = self.song_id_to_metada_path(song_id)
                with open_h5_file_read(metadata_path) as meta:
                    artist_terms = get_artist_terms(meta)
                    weights = get_artist_terms_freq(meta)

                    cluster_scores = self.calculate_cluster_scores(
                        artist_terms, weights
                    )
                    if len(cluster_scores) == 0:
                        continue

                    for cluster, score in cluster_scores.items():
                        if score > self._config["min_cluster_score"]:
                            songs_per_cluster[cluster] += 1
                            # logger.info(
                            #     f"Song matches cluster {best_cluster} confidently ({cluster_scores[best_cluster]:.2f})"
                            # )

                            # logger.info(f"Song artist: {get_artist_name(meta)}")
                            # logger.info(f"Song title: {get_title(meta)}")
                            # logger.info(f"Song artist terms: {get_artist_terms(meta)}")
                            # tags_in_cluster = [
                            #     self._cooccurrence_data["tag_list"][idx]
                            #     for idx in self._tag_clustering["labels_index"][
                            #         best_cluster
                            #     ]
                            # ]
                            # logger.info(f"Tags in cluster: {tags_in_cluster}")

        # Draw distribution
        plt.bar(list(range(len(songs_per_cluster))), songs_per_cluster)

        plt.title("Songs Cluster Distribution")
        plt.xlabel("Cluster Index")
        plt.ylabel("Songs Count")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self._root / "songs_cluster_distribution.png", dpi=300)
        plt.close()
