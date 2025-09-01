MIDI_VELOCITY_VOCAB_SIZE = 128


class VelocityQuantizer:
    """
    A class to quantize MIDI note velocities.
    """

    def __init__(self, velocity_bins: int = 32):
        """
        Initialize the VelocityQuantizer with a number of bins.

        :param velocity_bins: The number of velocity bins to quantize to.
        """
        self._velocity_bins = velocity_bins

    def quantize(self, velocity: int) -> int:
        """
        Quantize a single velocity value to the nearest bin.

        :param velocity: The velocity to quantize.
        :return: The quantized velocity bin index.
        """
        if velocity < 0:
            return 0
        elif velocity >= MIDI_VELOCITY_VOCAB_SIZE:
            return self._velocity_bins - 1
        else:
            return round((velocity / (MIDI_VELOCITY_VOCAB_SIZE - 1)) * (self._velocity_bins - 1))

    def velocity_bins(self) -> int:
        """
        Get the number of velocity bins.

        :return: The number of velocity bins.
        """
        return self._velocity_bins

    def velocity_bin_to_velocity(self, velocity_bin: int) -> int:
        return round((velocity_bin / (self._velocity_bins - 1)) * (MIDI_VELOCITY_VOCAB_SIZE - 1))
