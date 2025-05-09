import sys
from src.composition.midi.rhythm_quantizer import RhythmQuantizer
import pytest


def test_rhythm_quantizer():
    # Initialize the RhythmQuantizer with 4 ticks per beat (i.e., sixteenth notes)
    quantizer = RhythmQuantizer(ticks_per_beat=4)

    # Test quantization of a single beat duration
    beat_duration = 0.3
    snapped_duration, ticks = quantizer.quantize(beat_duration)
    assert snapped_duration == 0.25  # Closest bin is 0.25 (sixteenth)
    assert ticks == 1  # one tick


if __name__ == "__main__":
    sys.exit(pytest.main())
