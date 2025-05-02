import sys
from src.composition.midi.rhythm_quantizer import RhythmQuantizer
import pytest


def test_rhythm_quantizer():
    test_bins = [
        (0.5, "eighth"),
        (0.25, "sixteenth"),
        (0.125, "thirty_second"),
        (0.0625, "sixty_fourth"),
        (0.0, "null"),
    ]

    # Initialize the RhythmQuantizer
    quantizer = RhythmQuantizer(bins=test_bins)

    # Test quantization of a single beat duration
    beat_duration = 0.3
    snapped_duration, idx, label = quantizer.quantize(beat_duration)
    assert snapped_duration == 0.25  # Closest bin is 0.25 (sixteenth)
    assert idx == 1  # Index of the closest bin
    assert label == "sixteenth"  # Label of the closest bin

    # Test quantization of a sequence of beat durations
    beat_durations = [0.25, 0.25, 0.05, 0.04, 0.16, 0.5, 0.03, 0.47]
    snapped_durations, indices, labels = quantizer.quantize_sequence(beat_durations)
    assert snapped_durations == [
        0.25,
        0.25,
        0.0625,
        0.0625,
        0.125,
        0.5,
        0.0,
        0.5,
    ]  # Expected snapped durations
    assert indices == [1, 1, 3, 3, 2, 0, 4, 0]  # Expected indices of the closest bins
    assert labels == [
        "sixteenth",
        "sixteenth",
        "sixty_fourth",
        "sixty_fourth",
        "thirty_second",
        "eighth",
        "null",
        "eighth",
    ]  # Expected labels of the closest bins
    assert sum(snapped_durations) == sum(
        beat_durations
    )  # Check if total duration is preserved


if __name__ == "__main__":
    sys.exit(pytest.main())
