import torch
import math


class TempoNormalizer:
    def __init__(self, min_bpm=30, max_bpm=300):
        self.min_bpm_log = math.log(min_bpm)
        self.max_bpm_log = math.log(max_bpm)
        self.bpm_log_range = self.max_bpm_log - self.min_bpm_log

    def normalize_bpm(self, bpm):
        """
        Normalize BPM values to log scale.
        """
        bpm = torch.log(bpm.to(dtype=torch.float32))
        bpm = (bpm - self.min_bpm_log) / self.bpm_log_range
        return bpm

    def unnormalize_bpm(self, normalized_bpm):
        """
        Convert normalized BPM values back to original BPM scale.
        """
        normalized_bpm = normalized_bpm.to(dtype=torch.float32)
        return torch.exp(normalized_bpm * self.bpm_log_range + self.min_bpm_log)
