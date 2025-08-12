import pathlib
import sys
from src.composition.midi.velocity_quantizer import VelocityQuantizer
from src.composition.midi.tokenizer import MultiTrackMidiTokenizer
from src.composition.training.training_data.dataloader.utils import *

import pytest
from bazel_tools.tools.python.runfiles import runfiles

import numpy as np


def test_collate_fn():
    r = runfiles.Create()

    test_file_paths = [
        pathlib.Path(
            r.Rlocation(
                "_main/datasets/lakh/lmd_full/e/e5aeb1a3772e73b86c47542b9c5063b3.mid"
            )
        ),
        pathlib.Path(r.Rlocation("_main/datasets/vgmusic/xbox360/Elises_Tears.mid")),
        pathlib.Path(
            r.Rlocation("_main/datasets/vgmusic/xbox360/Dreams_of_an_Absolution.mid")
        ),
        pathlib.Path(r.Rlocation("_main/datasets/vgmusic/xbox360/halotheme.mid")),
        pathlib.Path(r.Rlocation("_main/datasets/vgmusic/xbox360/His_World_Remix.mid")),
    ]

    velocity_bins = 128

    velocity_quantizer = VelocityQuantizer(velocity_bins=velocity_bins)
    tokenizer = MultiTrackMidiTokenizer(
        velocity_quantizer=velocity_quantizer, max_instrument_instances=10
    )

    input_and_control_batch = []
    max_tracks = 0
    for test_file_path in test_file_paths:
        encode_result = tokenizer.encode(str(test_file_path))
        assert encode_result is not None, "Failed to encode"

        max_tracks = max(max_tracks, len(encode_result["tracks"]))

        # TODO: Add control input
        input_and_control_batch.append({"input": encode_result, "control": None})

    collated = collate_fn(input_and_control_batch)
    model_input_and_control = prepare_batch_for_model(collated)

    assert model_input_and_control is not None, "Failed to prepare batch for model"

    model_input = model_input_and_control["input"]

    # Per-song per-track sequences
    assert model_input["pitch_tokens"].shape == (
        len(test_file_paths),
        max_tracks,
        MAX_SEQ_LEN,
    ), "Pitch tokens shape mismatch"
    assert (
        model_input["velocity_tokens"].shape == model_input["pitch_tokens"].shape
    ), "Velocity tokens shape mismatch"
    assert (
        model_input["beat_positions"].shape == model_input["pitch_tokens"].shape
    ), "Beat positions shape mismatch"
    assert (
        model_input["bar_positions"].shape == model_input["pitch_tokens"].shape
    ), "Bar positions shape mismatch"
    assert (
        model_input["within_bar_positions"].shape == model_input["pitch_tokens"].shape
    ), "Within bar positions shape mismatch"
    assert (
        model_input["note_durations_beats"].shape == model_input["pitch_tokens"].shape
    ), "Note durations beats shape mismatch"
    assert (
        model_input["attention_mask"].shape == model_input["pitch_tokens"].shape
    ), "Attention mask shape mismatch"

    # Per-song per-track scalars
    assert model_input["track_mask"].shape == (
        len(test_file_paths),
        max_tracks,
    ), "Track mask shape mismatch"
    assert (
        model_input["program_ids"].shape == model_input["track_mask"].shape
    ), "Program IDs shape mismatch"

    # Per-song sequences
    assert model_input["bar_boundaries"].shape == (
        len(test_file_paths),
        MAX_BARS,
    ), "Bar boundaries shape mismatch"
    assert (
        model_input["bar_tempos"].shape == model_input["bar_boundaries"].shape
    ), "Bar tempos shape mismatch"
    assert (
        model_input["global_attention_mask"].shape
        == model_input["bar_boundaries"].shape
    ), "Global attention mask shape mismatch"

    for i in range(len(test_file_paths)):
        encoded = input_and_control_batch[i]["input"]
        model_input_i = {key: value[i] for key, value in model_input.items()}

        bar_boundaries_length = min(len(encoded["bar_boundaries"]), MAX_BARS)
        assert (
            model_input_i["bar_boundaries"][:bar_boundaries_length].numpy()
            == encoded["bar_boundaries"][:bar_boundaries_length]
        ).all(), f"Bar boundaries mismatch for song {i}"

        assert (
            model_input_i["bar_boundaries"][bar_boundaries_length:] == PAD_VALUE
        ).all(), f"Bar boundaries padding mismatch for song {i}"

        assert np.allclose(
            model_input_i["bar_tempos"][:bar_boundaries_length].numpy(),
            encoded["bar_tempos"][:bar_boundaries_length],
        ), f"Bar tempos mismatch for song {i}"

        assert (
            model_input_i["bar_tempos"][bar_boundaries_length:] == PAD_VALUE
        ).all(), f"Bar tempos padding mismatch for song {i}"

        assert (
            model_input_i["global_attention_mask"][:bar_boundaries_length].numpy() == 1
        ).all(), f"Global attention mask mismatch for song {i}"

        assert (
            model_input_i["global_attention_mask"][bar_boundaries_length:] == 0
        ).all(), f"Global attention mask padding mismatch for song {i}"

        for track_idx in range(max_tracks):
            if track_idx >= len(encoded["tracks"]):
                assert (
                    model_input_i["track_mask"][track_idx] == 0
                ), f"Track mask for song {i}, track {track_idx} should be 0"

                assert (
                    model_input_i["program_ids"][track_idx] == PAD_TRACK_ID
                ), f"Program ID for song {i}, track {track_idx} should be {PAD_TRACK_ID}"
                continue

            assert (
                model_input_i["track_mask"][track_idx] == 1
            ), f"Track mask for song {i}, track {track_idx} should be 1"

            track = encoded["tracks"][track_idx]
            expected_program_id = track["program_id"]

            assert (
                model_input_i["program_ids"][track_idx] == expected_program_id
            ), f"Program ID mismatch for song {i}, track {track_idx}"

            idx = bisect.bisect_left(track["bar_positions"], bar_boundaries_length)
            track_tokens_length = min(
                min(len(track["bar_positions"]), MAX_SEQ_LEN), idx
            )
            assert (
                model_input_i["pitch_tokens"][track_idx][:track_tokens_length].numpy()
                == track["pitch_tokens"][:track_tokens_length]
            ).all(), f"Pitch tokens mismatch for song {i}, track {track_idx}"

            assert (
                model_input_i["pitch_tokens"][track_idx][track_tokens_length:]
                == PAD_VALUE
            ).all(), f"Pitch tokens padding mismatch for song {i}, track {track_idx}"

            assert (
                model_input_i["velocity_tokens"][track_idx][
                    :track_tokens_length
                ].numpy()
                == track["velocity_tokens"][:track_tokens_length]
            ).all(), f"Velocity tokens mismatch for song {i}, track {track_idx}"

            assert (
                model_input_i["velocity_tokens"][track_idx][track_tokens_length:]
                == PAD_VALUE
            ).all(), f"Velocity tokens padding mismatch for song {i}, track {track_idx}"

            assert np.allclose(
                model_input_i["beat_positions"][track_idx][
                    :track_tokens_length
                ].numpy(),
                track["beat_positions"][:track_tokens_length],
            ), f"Beat positions mismatch for song {i}, track {track_idx}"

            assert (
                model_input_i["beat_positions"][track_idx][track_tokens_length:]
                == PAD_VALUE
            ).all(), f"Beat positions padding mismatch for song {i}, track {track_idx}"

            assert (
                model_input_i["bar_positions"][track_idx][:track_tokens_length].numpy()
                == track["bar_positions"][:track_tokens_length]
            ).all(), f"Bar positions mismatch for song {i}, track {track_idx}"

            assert (
                model_input_i["bar_positions"][track_idx][track_tokens_length:]
                == PAD_VALUE
            ).all(), f"Bar positions padding mismatch for song {i}, track {track_idx}"

            assert np.allclose(
                model_input_i["within_bar_positions"][track_idx][
                    :track_tokens_length
                ].numpy(),
                track["within_bar_positions"][:track_tokens_length],
            ), f"Within bar positions mismatch for song {i}, track {track_idx}"

            assert (
                model_input_i["within_bar_positions"][track_idx][track_tokens_length:]
                == PAD_VALUE
            ).all(), (
                f"Within bar positions padding mismatch for song {i}, track {track_idx}"
            )

            assert np.allclose(
                model_input_i["note_durations_beats"][track_idx][
                    :track_tokens_length
                ].numpy(),
                track["note_durations_beats"][:track_tokens_length],
            ), f"Note durations beats mismatch for song {i}, track {track_idx}"

            assert (
                model_input_i["note_durations_beats"][track_idx][track_tokens_length:]
                == PAD_VALUE
            ).all(), (
                f"Note durations beats padding mismatch for song {i}, track {track_idx}"
            )

            assert (
                model_input_i["attention_mask"][track_idx][:track_tokens_length].numpy()
                == 1
            ).all(), f"Attention mask mismatch for song {i}, track {track_idx}"

            assert (
                model_input_i["attention_mask"][track_idx][track_tokens_length:]
                == PAD_VALUE
            ).all(), f"Attention mask padding mismatch for song {i}, track {track_idx}"

            assert (
                model_input_i["bar_activations"][track_idx][
                    :bar_boundaries_length
                ].numpy()
                == track["bar_activations"][:bar_boundaries_length]
            ).all(), f"Bar activations mismatch for song {i}, track {track_idx}"

            assert (
                model_input_i["bar_activations"][track_idx][bar_boundaries_length:]
                == PAD_VALUE
            ).all(), f"Bar activations padding mismatch for song {i}, track {track_idx}"


if __name__ == "__main__":
    sys.exit(pytest.main())
