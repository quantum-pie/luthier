from collections import defaultdict
from typing import List, Tuple
import itertools


class RhythmQuantizer:
    """
    RhythmQuantizer is a class for quantizing beat durations into discrete
    rhythmic values. It uses a dynamic programming approach to minimize
    snapping error while preserving the total duration of a sequence of
    beat durations.
    """

    def __init__(self, ticks_per_beat=480):
        self.ticks_per_beat = ticks_per_beat

    def get_ticks_per_beat(self):
        """
        Get the number of ticks per beat.
        Returns:
            int: The number of ticks per beat.
        """
        return self.ticks_per_beat

    def quantize(self, beat_duration):
        """
        Quantize a single beat duration to the nearest value in the
        quantization bins.
        Args:
            beat_duration (float): The duration to quantize.
        Returns:
            Tuple of:
                - snapped duration (float)
                - class index/number of ticks (int)
        """
        quantized_ticks = round(beat_duration * self.ticks_per_beat)
        return quantized_ticks / self.ticks_per_beat, quantized_ticks
