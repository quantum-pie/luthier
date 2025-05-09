import bisect
import pathlib
import sys
from src.composition.midi.tokenizer import (
    DRUMS_PROGRAM_ID,
    MultiTrackMidiTokenizer,
)
from src.composition.midi.velocity_quantizer import VelocityQuantizer

import pytest
import pretty_midi
import numpy as np
from bazel_tools.tools.python.runfiles import runfiles


def beat_to_seconds(target_beat, tempo_beats, tempos):
    assert len(tempo_beats) == len(tempos), "tempo_beats and tempos must match"
    assert all(
        tempo_beats[i] <= tempo_beats[i + 1] for i in range(len(tempo_beats) - 1)
    ), "tempo_beats must be sorted"

    # Find the tempo region the target beat falls into
    tempo_index = bisect.bisect_right(tempo_beats, target_beat) - 1
    tempo_index = max(0, tempo_index)

    # Accumulate time up to the current tempo region
    seconds = 0.0
    for i in range(tempo_index):
        beat_span = tempo_beats[i + 1] - tempo_beats[i]
        seconds += (60.0 / tempos[i]) * beat_span

    # Add the partial time within the current region
    beat_offset = target_beat - tempo_beats[tempo_index]
    seconds += (60.0 / tempos[tempo_index]) * beat_offset

    return seconds


def reconstruct_pretty_midi_instrument_from_beat_timeline(
    program,
    is_drums,
    note_pitches,
    note_velocities,
    note_start_beats,
    note_lengths,
    tempo_beats,
    tempos,
):
    assert len(tempo_beats) == len(tempos), "Mismatch in beats and times"

    instrument = pretty_midi.Instrument(program=program, is_drum=is_drums)

    for pitch, velocity, start_beat, duration in zip(
        note_pitches, note_velocities, note_start_beats, note_lengths
    ):
        start_time = beat_to_seconds(start_beat, tempo_beats, tempos)
        end_time = beat_to_seconds(start_beat + duration, tempo_beats, tempos)

        instrument.notes.append(
            pretty_midi.Note(
                velocity=int(velocity), pitch=int(pitch), start=start_time, end=end_time
            )
        )

    return instrument


def test_tokenized_data_reconstruction():
    r = runfiles.Create()

    # Rare cases when PrettyMIDI doesn't fail and we can use it to compare
    # In many cases it fails to read tempo and time signature changes,
    # or parses drums incorrectly
    valid_test_file_paths = [
        pathlib.Path(r.Rlocation(location))
        for location in [
            "_main/datasets/lakh/lmd_full/c/c90f24428708b1bdfde935c698f37032.mid",
            "_main/datasets/lakh/lmd_full/3/3c0e36759a9e11695156bb6643f8fe32.mid",
            "_main/datasets/lakh/lmd_full/f/fd513cf270fe876c1c7218c72767518c.mid",
            "_main/datasets/vgmusic/xbox360/Elises_Tears.mid",
        ]
    ]

    invalid_test_file_paths = [
        pathlib.Path(r.Rlocation(location))
        for location in [
            "_main/datasets/lakh/lmd_full/d/d142de24c32fd5e117bf9d0c2f0e94cd.mid",
            "_main/datasets/lakh/lmd_full/f/fba0fdf6553c7ec76b1c5ed8d2d92642.mid",
        ]
    ]

    velocity_bins = 128
    velocity_quantizer = VelocityQuantizer(velocity_bins=velocity_bins)
    tokenizer = MultiTrackMidiTokenizer(
        velocity_quantizer=velocity_quantizer, max_instrument_instances=10
    )

    for test_file_path in invalid_test_file_paths:
        encode_result = tokenizer.encode(str(test_file_path))
        assert encode_result is None, "Didn't flag invalid MIDI file"

    for test_file_path in valid_test_file_paths:
        pm_gt = pretty_midi.PrettyMIDI(str(test_file_path))

        encode_result = tokenizer.encode(str(test_file_path))
        assert encode_result is not None, "Failed to encode"

        track_grouped_data = encode_result["tracks"]

        # reassemble midi from tokenized data
        pm_out = pretty_midi.PrettyMIDI(initial_tempo=encode_result["all_tempos"][0])

        tempos = encode_result["all_tempos"]
        tempo_beats = encode_result["tempo_change_beats"]

        for tokens in track_grouped_data:
            program_id = tokens["program_id"]
            pitches = tokens["pitch_tokens"]
            vel_tokens = tokens["velocity_tokens"]
            note_starts = tokens["beat_positions"]
            note_lengths = tokens["note_durations_beats"]

            is_drums = False
            if program_id == DRUMS_PROGRAM_ID:
                is_drums = True
                program_id = 0

            velocities = [
                velocity_quantizer.velocity_bin_to_velocity(v) for v in vel_tokens
            ]

            instrument = reconstruct_pretty_midi_instrument_from_beat_timeline(
                program_id,
                is_drums,
                pitches,
                velocities,
                note_starts,
                note_lengths,
                tempo_beats,
                tempos,
            )

            pm_out.instruments.append(instrument)

        # Save the reconstructed MIDI file for human inspection
        pm_out.write(f"/tmp/{test_file_path.stem}.mid")
        pm_out_test = pretty_midi.PrettyMIDI(f"/tmp/{test_file_path.stem}.mid")

        # Check if the reconstructed MIDI file is similar to the original
        for gt_inst, out_inst in zip(pm_gt.instruments, pm_out_test.instruments):
            assert gt_inst.program == out_inst.program, "Program ID mismatch"

            gt_notes = sorted(
                gt_inst.notes, key=lambda note: (note.start, note.end, note.pitch)
            )
            out_notes = sorted(
                out_inst.notes, key=lambda note: (note.start, note.end, note.pitch)
            )

            assert len(gt_inst.notes) == len(out_inst.notes), "Number of notes mismatch"
            for gt_note, out_note in zip(gt_notes, out_notes):
                assert (
                    gt_note.pitch == out_note.pitch
                ), f"Pitch mismatch: {gt_note.pitch} != {out_note.pitch}"
                assert gt_note.start == pytest.approx(
                    out_note.start, rel=1e-3
                ), f"Start time mismatch: {gt_note.start} != {out_note.start}"
                assert gt_note.end == pytest.approx(
                    out_note.end, rel=1e-3
                ), f"End time mismatch: {gt_note.end} != {out_note.end}"


if __name__ == "__main__":
    sys.exit(pytest.main())
