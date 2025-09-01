import torch.nn.functional as F
import torch

from src.composition.midi.tempo_normalizer import TempoNormalizer


def tempo_loss(normalized_tempos_pred, normalized_tempos_target):
    """
    Compute the loss for tempo prediction.
    Args:
        normalized_tempos_pred: FloatTensor of shape [B, T] — predicted normalized tempos
        normalized_tempos_target: FloatTensor of shape [B, T] — target normalized tempos
    Returns:
        FloatTensor of shape [B, T] — tempo loss
    """
    # Compute the L1 loss between predicted and target tempos
    return F.smooth_l1_loss(normalized_tempos_pred, normalized_tempos_target, reduction="none")


def generate_tempo_targets(tempos_bpm):
    """
    Generate targets for tempo prediction.
    Args:
        tempos: FloatTensor of shape [B, T] — raw tempos in BPM
    Returns:
        FloatTensor of shape [B, T] — normalized tempos
    """
    # Normalize the tempos to a range suitable for training
    return TempoNormalizer().normalize_bpm(tempos_bpm)
