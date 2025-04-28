import os
import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging
from pathlib import Path
from tqdm import tqdm
import click

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
TIMEOUT_PAGE = 15
TIMEOUT_FILE = 30

# Headers to avoid basic bot detection
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
}


def scrape_source(source_url, source_name, dataset_dir):
    logger.info(f"Scraping source page: {source_url}")
    try:
        response = requests.get(source_url, headers=HEADERS, timeout=TIMEOUT_PAGE)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to fetch source page {source_url}: {e}")
        return

    soup = BeautifulSoup(response.text, "html.parser")

    source_folder = dataset_dir / source_name
    os.makedirs(source_folder, exist_ok=True)

    current_game_name = "Unknown"

    for tr in soup.find_all("tr"):
        if "gameheader" in tr.get("class", []):
            td = tr.find("td", class_="header")
            if td:
                current_game_name = td.get_text(strip=True)
        else:
            a_tag = tr.find("a", href=True)
            if a_tag and (
                a_tag["href"].endswith(".mid") or a_tag["href"].endswith(".midi")
            ):
                file_url = urljoin(source_url, a_tag["href"])
                filename = os.path.basename(urlparse(file_url).path)
                song_name = a_tag.get_text(strip=True)

                save_midi_and_metadata(
                    file_url, filename, current_game_name, song_name, source_folder
                )


def save_midi_and_metadata(file_url, filename, game_name, song_name, folder):
    midi_path = os.path.join(folder, filename)
    json_path = midi_path.rsplit(".", 1)[0] + ".json"

    # Skip if already downloaded
    if os.path.exists(midi_path) and os.path.exists(json_path):
        logger.info(f"Already exists, skipping: {midi_path}")
        return

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(file_url, headers=HEADERS, timeout=TIMEOUT_FILE)
            response.raise_for_status()
            with open(midi_path, "wb") as f:
                f.write(response.content)
            logger.info(f"Downloaded: {midi_path}")

            metadata = {"game_name": game_name, "song_name": song_name}
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved metadata: {json_path}")
            return  # Success

        except requests.exceptions.Timeout:
            logger.warning(
                f"Timeout downloading {file_url}, attempt {attempt}/{MAX_RETRIES}"
            )
        except Exception as e:
            logger.error(
                f"Error downloading {file_url}: {e}, attempt {attempt}/{MAX_RETRIES}"
            )

        time.sleep(2)  # Wait a bit before retrying

    logger.error(f"Failed to download after {MAX_RETRIES} attempts: {file_url}")


@click.command()
@click.option("--dataset_dir", type=Path, required=True)
@click.option("--sources_file", type=Path, required=True)
def cli(dataset_dir, sources_file):
    if not Path.exists(dataset_dir):
        logger.error(f"Dataset directory '{dataset_dir}' does not exist.")
        return

    if not Path.exists(sources_file):
        logger.error(f"Sources file '{sources_file}' does not exist.")
        return

    with open(sources_file, "r", encoding="utf-8") as f:
        source_links = [line.strip() for line in f if line.strip()]

    for source_url in source_links:
        source_name = source_url.strip("/").split("/")[-1]
        scrape_source(source_url, source_name, dataset_dir)

    logger.info("Scraping complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
