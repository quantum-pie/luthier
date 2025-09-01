from pathlib import Path
import click
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

from sentence_transformers import SentenceTransformer


from src.composition.training.training_data.lakh.hdf5_getters import *
from bazel_tools.tools.python.runfiles import runfiles

from src.composition.training.training_data.lakh.lakh import Lakh

logger = logging.getLogger(__name__)


class MatchedMidiType(Enum):
    matched = (0,)
    aligned = 1


class TagClusterer:
    """
    This class is responsible for clustering the aligned
    Lakh MIDI dataset tags and assigning tags to song ids. Note that the clustering
    result is not usable in commercial product, because tags are coming from Eagle Nest API.
    The clustering is done using HDBSCAN and UMAP.
    """

    def __init__(self, data_dir: Path):
        r = runfiles.Create()
        config_path = Path(r.Rlocation("_main/src/composition/training/training_data/lakh/tag_clustering_config.yaml"))

        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)["tag_clustering"]

        self._root = data_dir
        self._lakh = Lakh(data_dir)
        self._cooccurrence_path = data_dir / "cooccurrence.pkl"
        self._cooccurrence_data = {}
        self._tag_clustering_path = data_dir / "clustering.pkl"
        self._tag_clustering = {}

        self._sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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

        filtered_cooccurrence = self._cooccurrence_data["tag_cooccurrence"][np.ix_(valid_indices, valid_indices)]

        # Step 2: Normalize tag co-occurrence vectors
        X_norm = normalize(filtered_cooccurrence.astype(float), norm="l2", axis=1)

        # Step 3: Reduce dimensionaly
        reducer = umap.UMAP(
            n_neighbors=10,
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
            tag_names = [self._cooccurrence_data["tag_list"][i] for i in tag_indices]

            embeddings = self._sentence_model.encode(tag_names)
            centroid = np.mean(embeddings, axis=0)

            similarities = cosine_similarity([centroid], embeddings)[0]
            ranked_indices = np.argsort(similarities)[::-1]
            top_tokens = [tag_names[i] for i in ranked_indices[: min(3, len(tag_names))]]
            cluster_name = f"{top_tokens[0]} ({' / '.join(top_tokens[1:])})"

            labels_names[int(label)] = cluster_name
            logger.info(f"Cluster id {label} has name {cluster_name}")
            logger.info(f"Cluster id {label} has tags {tag_names}")

        result = {
            "labels_list": labels,
            "labels_index": labels_index,
            "labels_names": labels_names,
        }
        self._tag_clustering = result
        with open(self._tag_clustering_path, "wb") as f:
            pickle.dump(result, f)

        with open(self._root / "cluster_id_to_name.yaml", "w") as f:
            yaml.dump({"clusted_id_to_name": labels_names}, f)

    def build_tag_cooccurrence(self):
        if Path.exists(self._cooccurrence_path):
            logger.info("Found cached co-occurrence data!")
            with open(self._cooccurrence_path, "rb") as f:
                self._cooccurrence_data = pickle.load(f)
                return

        tag_set = set()
        logger.info("Collecting tags...")
        for song_id, matches in tqdm(self._lakh.get_match_scores().items()):
            for _, score in matches.items():
                if score < self._config["min_match_score"]:
                    continue

                metadata_path = self._lakh.song_id_to_metada_path(song_id)
                with open_h5_file_read(metadata_path) as meta:
                    artist_terms = get_artist_terms(meta)
                    tag_set.update(tag.decode("utf-8").lower() for tag in artist_terms)

        tag_list = list(tag_set)
        tag_index = {tag: i for i, tag in enumerate(tag_list)}

        logger.info(f"Found {len(tag_index)} unique tags.")

        tag_counts = np.zeros(len(tag_list), np.uint32)
        tag_cooccurrence = np.zeros((len(tag_list), len(tag_list)), np.uint32)

        logger.info("Calculating co-occurrence...")
        for song_id, matches in tqdm(self._lakh.get_match_scores().items()):
            for _, score in matches.items():
                if score < self._config["min_match_score"]:
                    continue

                metadata_path = self._lakh.song_id_to_metada_path(song_id)
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
        songs_per_cluster = np.zeros(len(self._tag_clustering["labels_index"]), np.uint32)
        for song_id, matches in tqdm(self._lakh.get_match_scores().items()):
            for md5, score in matches.items():
                if score < self._config["min_match_score"]:
                    continue

                metadata_path = self._lakh.song_id_to_metada_path(song_id)
                with open_h5_file_read(metadata_path) as meta:
                    artist_terms = get_artist_terms(meta)
                    weights = get_artist_terms_freq(meta)

                    cluster_scores = self.calculate_cluster_scores(artist_terms, weights)
                    if len(cluster_scores) == 0:
                        continue

                    for cluster, score in cluster_scores.items():
                        if score > self._config["min_cluster_score"]:
                            songs_per_cluster[cluster] += 1

        # Draw distribution
        plt.bar(list(range(len(songs_per_cluster))), songs_per_cluster)

        plt.title("Songs Cluster Distribution")
        plt.xlabel("Cluster Index")
        plt.ylabel("Songs Count")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self._root / "songs_cluster_distribution.png", dpi=300)
        plt.close()


@click.command()
@click.option("--dataset_dir", type=Path, required=True)
def cli(dataset_dir):
    clusterer = TagClusterer(dataset_dir)
    clusterer.build_tag_cooccurrence()
    clusterer.cluster_tags()
    clusterer.classify_songs_into_tag_clusters()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
