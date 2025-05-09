import torch


class TempoNormalizer:
    def __init__(self, min_bpm=30, max_bpm=300):
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.range = max_bpm - min_bpm

    def normalize_bpm(self, bpm):
        """
        Normalize BPM values to the [0, 1] range using PyTorch ops.
        Input can be a scalar or tensor. Returns tensor.
        """
        bpm = bpm.to(dtype=torch.float32)
        norm = (bpm - self.min_bpm) / self.range
        return torch.clamp(norm, 0.0, 1.0)

    def unnormalize_bpm(self, normalized_bpm):
        """
        Convert normalized BPM values back to original BPM scale.
        """
        normalized_bpm = normalized_bpm.to(dtype=torch.float32)
        return normalized_bpm * self.range + self.min_bpm
