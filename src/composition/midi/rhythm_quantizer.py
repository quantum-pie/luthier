from collections import defaultdict
from typing import List, Tuple


class RhythmQuantizer:
    """
    RhythmQuantizer is a class for quantizing beat durations into discrete
    rhythmic values. It uses a dynamic programming approach to minimize
    snapping error while preserving the total duration of a sequence of
    beat durations.
    """

    def __init__(self, bins=None):
        # Default: list of (beat_value, human_label)
        default_bins = [
            # Binary values
            (4.0, "whole"),
            (2.0, "half"),
            (1.0, "quarter"),
            (0.5, "eighth"),
            (0.25, "sixteenth"),
            (0.125, "thirty_second"),
            # Dotted values
            (3.0, "dotted_half"),
            (1.5, "dotted_quarter"),
            (0.75, "dotted_eighth"),
            (0.375, "dotted_sixteenth"),
            (0.1875, "dotted_thirty_second"),
            # Triplets
            (2 / 3, "half_triplet"),
            (1 / 3, "quarter_triplet"),
            (1 / 6, "eighth_triplet"),
            # Quintuplets
            (0.8, "half_quintuplet"),
            (0.4, "quarter_quintuplet"),
            (0.2, "eighth_quintuplet"),
            # Septuplets
            (4 / 7, "half_septuplet"),
            (2 / 7, "quarter_septuplet"),
            (1 / 7, "eighth_septuplet"),
            # Rest / no-op
            (0.0, "null"),
        ]

        if bins is None:
            bins = default_bins
        self.bins = bins

        self.values = [b[0] for b in self.bins]
        self.labels = [b[1] for b in self.bins]
        self.value_to_idx = {round(v, 6): i for i, v in enumerate(self.values)}
        self.rounding_precision = 6

    def quantize(self, beat_duration):
        """
        Quantize a single beat duration to the nearest value in the
        quantization bins.
        Args:
            beat_duration (float): The duration to quantize.
        Returns:
            Tuple of:
                - snapped duration (float)
                - class index (int)
                - human-readable label (str)
        """
        diffs = [abs(beat_duration - b) for b in self.values]
        best_idx = diffs.index(min(diffs))
        return self.values[best_idx], best_idx, self.labels[best_idx]

    def quantize_sequence(
        self, beat_durations: List[float]
    ) -> Tuple[List[float], List[int], List[str]]:
        """
        Quantize a sequence of beat durations using dynamic programming,
        preserving total duration and minimizing total snapping error.

        Note that this method assumes that the sum of the beat_durations
        can be exactly represented as a sum of the quantization bins.
        If this is not the case, a ValueError will be raised.

        Args:
            beat_durations (List[float]): List of beat durations to quantize.

        Returns:
            Tuple of:
                - snapped durations (List[float])
                - class indices (List[int])
                - human-readable labels (List[str])
        """
        n = len(beat_durations)
        total = round(sum(beat_durations), self.rounding_precision)

        dp = [defaultdict(lambda: (float("inf"), [])) for _ in range(n + 1)]
        dp[0][0.0] = (0.0, [])

        for i in range(n):
            for acc_sum, (err, path) in dp[i].items():
                for val in self.values:
                    new_sum = acc_sum + val
                    new_err = err + abs(val - beat_durations[i])
                    if new_err < dp[i + 1][new_sum][0]:
                        dp[i + 1][new_sum] = (new_err, path + [val])

        best_path = None
        for total_sum, (err, path) in dp[n].items():
            if round(total_sum, self.rounding_precision) == total:
                best_path = path
                break

        if best_path is None:
            raise ValueError("No valid quantization path found.")

        indices = [
            self.value_to_idx[round(v, self.rounding_precision)] for v in best_path
        ]
        labels = [self.labels[i] for i in indices]

        return best_path, indices, labels

    def class_value(self, class_idx):
        """
        Get the quantization value for a given class index.
        Args:
            class_idx (int): The index of the class.
        Returns:
            float: The quantization value for the class.
        """
        return self.values[class_idx]

    def class_label(self, class_idx):
        """
        Get the human-readable label for a given class index.
        Args:
            class_idx (int): The index of the class.
        Returns:
            str: The human-readable label for the class.
        """
        return self.labels[class_idx]

    def num_classes(self):
        """
        Get the number of quantization classes.
        Returns:
            int: The number of quantization classes.
        """
        return len(self.values)
