from collections import defaultdict
import bisect
from mido import MidiFile, tempo2bpm
import logging

from fractions import Fraction


DRUMS_PROGRAM_ID = 128
MAX_TICK = 1e7

logger = logging.getLogger(__name__)


class MultiTrackMidiTokenizer:
    def __init__(self, velocity_quantizer, max_instrument_instances):
        self.velocity_quantizer = velocity_quantizer
        self.max_instrument_instances = max_instrument_instances

    @staticmethod
    def parse_tempo_map(mid):
        tempo_events = {}
        for track in mid.tracks:
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time
                if msg.type == "set_tempo":
                    tempo_events[abs_tick] = tempo2bpm(msg.tempo)

        if len(tempo_events) == 0:
            logger.warning("The file has no tempo messages, using default tempo")
            tempo_events = {0: 120.0}  # Default tempo if none is found

        tempo_times, tempi = zip(
            *sorted(tempo_events.items(), key=lambda item: item[0])
        )

        if tempo_times[0] != 0:
            logger.error("Tempo is not set at the beginning of the file")
            return None, None

        return list(tempo_times), list(tempi)

    @staticmethod
    def parse_time_signatures(mid):
        time_sigs = {}
        for track in mid.tracks:
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time
                if msg.type == "time_signature":
                    time_sigs[abs_tick] = (msg.numerator, msg.denominator)

        if len(time_sigs) == 0:
            logger.warning(
                "The file has no time signature messages, using default time signature"
            )
            time_sigs = {0: (4, 4)}  # Default time signature if none is found

        ts_times, time_signatures = zip(
            *sorted(time_sigs.items(), key=lambda item: item[0])
        )

        if ts_times[0] != 0:
            logger.error("Time signature is not set at the beginning of the file")
            return None, None

        return list(ts_times), list(time_signatures)

    @staticmethod
    def generate_beat_ticks(first_tick, last_tick, ticks_per_beat):
        return list(range(first_tick, last_tick, ticks_per_beat))

    @staticmethod
    def compute_bar_boundaries_from_beats(
        beat_ticks, ts_times, ts_signatures, ticks_per_beat
    ):
        assert len(ts_times) == len(
            ts_signatures
        ), "Time signature and time list must match"
        if not beat_ticks:
            return []

        bar_boundaries = []
        ts_index = 0
        current_tick = beat_ticks[0]  # Start from first beat

        while current_tick <= beat_ticks[-1]:
            # Update time signature if needed
            while (
                ts_index + 1 < len(ts_times) and current_tick >= ts_times[ts_index + 1]
            ):
                ts_index += 1

            numerator, denominator = ts_signatures[ts_index]
            beats_per_bar = (
                Fraction(numerator, denominator) * 4
            )  # quarter note equivalents
            bar_duration_ticks = beats_per_bar * ticks_per_beat

            bar_boundaries.append(int(current_tick))
            current_tick += bar_duration_ticks

        # Add one more final bar if the last bar end < last beat
        if bar_boundaries:
            last_bar_start = bar_boundaries[-1]
            numerator, denominator = ts_signatures[-1]
            bar_duration_ticks = Fraction(numerator, denominator) * 4 * ticks_per_beat
            last_bar_end = last_bar_start + bar_duration_ticks

            if last_bar_end > beat_ticks[-1]:
                bar_boundaries.append(float(last_bar_end))

        return bar_boundaries

    @staticmethod
    def tempo_at_time(tempo_times, tempi, beat_time):
        i = bisect.bisect_right(tempo_times, beat_time) - 1
        if i < 0:
            return tempi[0]
        return tempi[i]

    @staticmethod
    def bar_position_from_tick(bar_boundaries, beat_time):
        i = bisect.bisect_right(bar_boundaries, beat_time) - 1
        if i < 0 or i >= len(bar_boundaries) - 1:
            return 0, 0.0

        bar_start = bar_boundaries[i]
        bar_end = bar_boundaries[i + 1]
        return i, (beat_time - bar_start) / (bar_end - bar_start + 1e-9)

    def encode(self, midi_file_path):
        try:
            mid = MidiFile(midi_file_path)
            max_tick = max(
                msg.time
                for track in mid.tracks
                for msg in track
                if hasattr(msg, "time")
            )
            if max_tick > MAX_TICK:
                logger.error(f"MIDI file {midi_file_path} exceeds maximum tick limit")
                return None
        except Exception as e:
            logger.error(f"Failed to read MIDI file {midi_file_path}: {e}")
            return None

        tempo_times, tempi = MultiTrackMidiTokenizer.parse_tempo_map(mid)
        if tempo_times is None or tempi is None:
            return None

        ts_times, time_signatures = MultiTrackMidiTokenizer.parse_time_signatures(mid)
        if ts_times is None or time_signatures is None:
            return None

        notes_by_group = defaultdict(list)
        for track in mid.tracks:
            abs_tick = 0
            note_stack = defaultdict(lambda: defaultdict(list))
            program_by_channel = {
                i: 0 for i in range(16)
            }  # Default program for each channel
            program_by_channel[9] = DRUMS_PROGRAM_ID  # Drums channel
            for msg in track:
                abs_tick += msg.time
                if msg.type == "program_change" and msg.channel != 9:
                    program_by_channel[msg.channel] = msg.program
                elif msg.type == "note_on" and msg.velocity > 0:
                    stack_group = msg.channel
                    note_stack[stack_group][msg.note].append((abs_tick, msg.velocity))
                elif msg.type == "note_off" or (
                    msg.type == "note_on" and msg.velocity == 0
                ):
                    if msg.channel not in program_by_channel:
                        logger.error("Program is not set for channel!")
                        return None

                    prog = program_by_channel[msg.channel]
                    group = (msg.channel, prog)

                    stack_group = msg.channel

                    if (
                        msg.note in note_stack[stack_group]
                        and note_stack[stack_group][msg.note]
                    ):
                        notes_to_close = [
                            note
                            for note in note_stack[stack_group][msg.note]
                            if note[0] != abs_tick
                        ]

                        notes_to_keep = [
                            note
                            for note in note_stack[stack_group][msg.note]
                            if note[0] == abs_tick
                        ]

                        for start_time, velocity in notes_to_close:
                            notes_by_group[group].append(
                                (msg.note, velocity, start_time, abs_tick)
                            )

                        note_stack[stack_group][msg.note] = notes_to_keep

        notes_by_track = [
            (program, notes) for (_, program), notes in notes_by_group.items() if notes
        ]

        if not notes_by_track:
            logger.warning("No notes found in the MIDI file")
            return None

        all_note_ends = set()
        for _, notes in notes_by_track:
            all_note_ends.update(note[3] for note in notes)
        all_note_ends = sorted(all_note_ends)

        latest_ts = max([all_note_ends[-1], tempo_times[-1], ts_times[-1]])
        beats = MultiTrackMidiTokenizer.generate_beat_ticks(
            0, latest_ts + mid.ticks_per_beat, mid.ticks_per_beat
        )

        bar_boundaries = MultiTrackMidiTokenizer.compute_bar_boundaries_from_beats(
            beats, ts_times, time_signatures, mid.ticks_per_beat
        )

        # data on a fixed per-bar grid for latent network
        tempos_on_bars = [
            MultiTrackMidiTokenizer.tempo_at_time(tempo_times, tempi, beat)
            for beat in bar_boundaries
        ]

        track_grouped_data = []
        same_program_id_counts = defaultdict(int)
        for program_id, notes in notes_by_track:
            same_program_id_counts[program_id] += 1
            if same_program_id_counts[program_id] > self.max_instrument_instances:
                logger.warning(
                    f"Too many instances of program {program_id} ({same_program_id_counts[program_id]}), skipping"
                )
                continue

            new_data_list = []

            bar_activations = [False] * len(
                bar_boundaries
            )  # Initialize list of booleans
            for note in notes:
                start_ticks = note[2]
                start_beats = start_ticks / mid.ticks_per_beat
                end_ticks = note[3]
                end_beats = end_ticks / mid.ticks_per_beat

                duration_beats = end_beats - start_beats

                bar_position, within_bar_position = (
                    MultiTrackMidiTokenizer.bar_position_from_tick(
                        bar_boundaries, start_ticks
                    )
                )

                velocity_bin = self.velocity_quantizer.quantize(note[1])

                # Find bar indices this note overlaps with
                bar_start_idx = bisect.bisect_right(bar_boundaries, start_ticks) - 1
                bar_end_idx = bisect.bisect_right(bar_boundaries, end_ticks) - 1

                bar_start_idx = max(0, bar_start_idx)
                bar_end_idx = max(0, bar_end_idx)

                for i in range(bar_start_idx, bar_end_idx + 1):
                    bar_activations[i] = True

                new_data_list.append(
                    {
                        "pitch_token": note[0],
                        "velocity_token": velocity_bin,
                        "beat_position": start_beats,
                        "note_duration_beats": duration_beats,
                        "bar_position": bar_position,
                        "within_bar_position": within_bar_position,
                    }
                )

            new_data_list = sorted(new_data_list, key=lambda d: d["beat_position"])
            new_data = {
                "program_id": program_id,
                "pitch_tokens": [d["pitch_token"] for d in new_data_list],
                "velocity_tokens": [d["velocity_token"] for d in new_data_list],
                "beat_positions": [d["beat_position"] for d in new_data_list],
                "bar_positions": [d["bar_position"] for d in new_data_list],
                "within_bar_positions": [
                    d["within_bar_position"] for d in new_data_list
                ],
                "note_durations_beats": [
                    d["note_duration_beats"] for d in new_data_list
                ],
                "bar_activations": bar_activations,
            }

            # sort by beat_position here
            track_grouped_data.append(new_data)

        return {
            "tracks": track_grouped_data,
            "bar_tempos": tempos_on_bars,
            "all_tempos": tempi,
            "bar_boundaries": [b / mid.ticks_per_beat for b in bar_boundaries],
            "tempo_change_beats": [t / mid.ticks_per_beat for t in tempo_times],
        }
