from concurrent.futures import ProcessPoolExecutor, as_completed
import click
import mido
import multiprocessing as mp
import logging
from tqdm import tqdm
from bazel_tools.tools.python.runfiles import runfiles

logger = logging.getLogger(__name__)

TIMEOUT = 1  # seconds for each MIDI file to be processed


def load_midi(path):
    try:
        mido.MidiFile(path)
        return (path, "mido", None)
    except Exception as e:
        return (path, None, str(e))


def validate_all_parallel(paths, max_workers=8):
    valid = []
    errors = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_midi, path): path for path in paths}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Validating MIDI files"):
            path = futures[future]
            try:
                result = future.result(timeout=TIMEOUT)
                path, backend, err = result
                if backend:
                    valid.append(path)
                else:
                    errors.append((path, err))
            except TimeoutError:
                errors.append((path, "timeout"))

    return valid, errors


@click.command()
@click.option("--output_manifest", required=True, type=click.Path())
def filter_valid_midi_files(output_manifest):
    """
    Filter valid MIDI files from the input manifest and write them to the output manifest.
    """
    r = runfiles.Create()

    all_paths = []

    input_manifest = r.Rlocation("_main/datasets/lakh/lmd_full_manifest.txt")
    with open(input_manifest) as f:
        all_paths = [r.Rlocation(f"_main/{rel_path}") for rel_path in f.read().splitlines()]

    valid_paths, error_log = validate_all_parallel(all_paths)

    for path in valid_paths:
        logger.info(f"Valid: {path}")

    for path, err in error_log:
        logger.info(f"Invalid: {path} — {err}")

    with open(output_manifest, "w") as outfile:
        outfile.write("\n".join(valid_paths))

    logger.info(f"Valid MIDI files written to {output_manifest}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    filter_valid_midi_files()
