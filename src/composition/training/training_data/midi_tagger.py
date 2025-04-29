from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv
import os
import json
import requests
from urllib.parse import quote

import click
import re

from src.composition.game_genres import GameGenres
from src.composition.game_moods import GameMoods

from src.composition.training.training_data.midi_player import MidiPlayer


def clean_game_name(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


class MidiTaggerApp:
    def __init__(self, root, soundfont_path=None):
        self.player = MidiPlayer(soundfont_path)

        self.root = root
        self.root.title("MIDI Tagger with Metadata")

        self.file_list = []
        self.current_index = 0

        self.folder = None
        self.descriptions_cache = {}

        self.create_widgets()

    def create_widgets(self):
        frame = tk.Frame(self.root)
        frame.pack(padx=10, pady=10)

        # Filename label
        self.filename_label = tk.Label(frame, text="No file loaded", width=60)
        self.filename_label.pack(pady=5)

        # Metadata display
        self.metadata_label = tk.Label(
            frame, text="", width=60, justify="left", anchor="w"
        )
        self.metadata_label.pack(pady=5)

        # Game description
        self.description_label = tk.Text(frame, wrap=tk.WORD, height=8, width=80)
        self.description_label.configure(state="disabled")
        self.description_label.pack(pady=5)

        listboxes_frame = tk.Frame(frame)
        listboxes_frame.pack(pady=10)

        # Genre dropdown
        tk.Label(listboxes_frame, text="Genres:").grid(row=0, column=0)
        self.genre_listbox = tk.Listbox(
            listboxes_frame, selectmode="multiple", exportselection=False, height=20
        )
        for g in GameGenres:
            self.genre_listbox.insert(tk.END, g.value)
        self.genre_listbox.grid(row=1, column=0, padx=5, pady=5)

        # Mood dropdown
        tk.Label(listboxes_frame, text="Moods:").grid(row=0, column=1)
        self.mood_listbox = tk.Listbox(
            listboxes_frame, selectmode="multiple", exportselection=False, height=20
        )
        for m in GameMoods:
            self.mood_listbox.insert(tk.END, m.value)
        self.mood_listbox.grid(row=1, column=1, padx=5, pady=5)

        # Buttons
        button_frame = tk.Frame(frame)
        button_frame.pack(pady=10)

        self.play_button = tk.Button(button_frame, text="Play", command=self.play_midi)
        self.play_button.grid(row=0, column=0, padx=5)

        self.stop_button = tk.Button(
            button_frame, text="Stop Music", command=self.stop_midi
        )
        self.stop_button.grid(row=0, column=1, padx=5)

        self.keep_button = tk.Button(
            button_frame, text="Keep", command=lambda: self.save_tag(True)
        )
        self.keep_button.grid(row=0, column=2, padx=5)

        self.discard_button = tk.Button(
            button_frame, text="Discard", command=lambda: self.save_tag(False)
        )
        self.discard_button.grid(row=0, column=3, padx=5)

        self.load_button = tk.Button(
            frame, text="Load MIDI Folder", command=self.load_folder
        )
        self.load_button.pack(pady=5)

    def load_folder(self):
        self.folder = filedialog.askdirectory()
        if self.folder:
            description_cache_path = Path(self.folder) / "descriptions_cache.json"
            if Path.exists(description_cache_path):
                with open(description_cache_path, "r", encoding="utf-8") as f:
                    self.descriptions_cache = json.load(f)
            else:
                self.descriptions_cache = {}

            all_files = [
                str(Path(root_dir, file).as_posix())
                for root_dir, _, files in os.walk(self.folder)
                for file in files
                if file.endswith((".mid", ".midi"))
            ]

            already_tagged = set()
            if os.path.exists("tags.csv"):
                with open("tags.csv", newline="") as f:
                    reader = csv.reader(f)
                    next(reader, None)  # skip header if present
                    for row in reader:
                        if row:
                            already_tagged.add(row[0])

            self.file_list = [
                f for f in all_files if os.path.basename(f) not in already_tagged
            ]
            self.current_index = 0
            if self.file_list:
                self.load_current_file()
            else:
                messagebox.showinfo("All Done", "All files already tagged!")

    def fetch_game_description(self, game_name):
        if game_name in self.descriptions_cache:
            return self.descriptions_cache[game_name]

        # Clean special characters
        safe_name = clean_game_name(game_name)

        # Fetch description from Wikipedia API
        words = safe_name.split("_")

        while len(words) > 1:
            safe_name = "_".join(words)
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_name}"
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                description = data.get("extract", "No description available.")
                if description != "No description available.":
                    break
            except Exception:
                description = "No description available."

            words.pop()

        self.descriptions_cache[game_name] = description

        # Save updated cache
        description_cache_path = Path(self.folder) / "descriptions_cache.json"
        with open(description_cache_path, "w", encoding="utf-8") as f:
            json.dump(self.descriptions_cache, f, ensure_ascii=False, indent=2)

        return description

    def load_current_file(self):
        if self.current_index < len(self.file_list):
            filename = os.path.basename(self.file_list[self.current_index])
            filepath = self.file_list[self.current_index]
            self.filename_label.config(text=filename)

            metadata_text = ""
            metadata_path = filepath.rsplit(".", 1)[0] + ".json"
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    game = metadata.get("game_name", "Unknown Game")
                    song = metadata.get("song_name", "Unknown Song")
                    metadata_text = f"Game: {game}\nSong: {song}"

                    if game != "Unknown Game":
                        description = self.fetch_game_description(game)
                        self.description_label.configure(state="normal")
                        self.description_label.delete("1.0", tk.END)
                        self.description_label.insert(tk.END, description)
                        self.description_label.configure(state="disabled")

            else:
                metadata_text = "No metadata available"

            self.metadata_label.config(text=metadata_text)
        else:
            messagebox.showinfo("Done", "No more files to tag!")

    def play_midi(self):
        if self.current_index < len(self.file_list):
            self.player.play_midi(self.file_list[self.current_index])

    def stop_midi(self):
        self.player.stop_midi()

    def save_tag(self, keep):
        if self.current_index < len(self.file_list):
            filename = os.path.basename(self.file_list[self.current_index])
            selected_genres = [
                self.genre_listbox.get(i) for i in self.genre_listbox.curselection()
            ]
            selected_moods = [
                self.mood_listbox.get(i) for i in self.mood_listbox.curselection()
            ]

            if keep and (not selected_genres or not selected_moods):
                messagebox.showwarning(
                    "Missing Data", "Please select both genre and mood before keeping."
                )
                return

            tags_path = Path(self.folder) / "tags.csv"
            new_file = not Path.exists(tags_path)
            with open(tags_path, "a", newline="") as f:
                writer = csv.writer(f)
                if new_file:
                    writer.writerow(["filename", "genres", "moods", "keep"])
                if keep:
                    writer.writerow(
                        [
                            filename,
                            ";".join(selected_genres),
                            ";".join(selected_moods),
                            True,
                        ]
                    )
                else:
                    writer.writerow([filename, "", "", False])

            self.current_index += 1
            if self.current_index < len(self.file_list):
                self.genre_listbox.selection_clear(0, tk.END)
                self.mood_listbox.selection_clear(0, tk.END)
                self.load_current_file()
            else:
                messagebox.showinfo("Done", "All files tagged!")

    def on_closing(self):
        self.stop_midi()
        self.root.destroy()


@click.command()
@click.option(
    "--sf2", type=click.Path(exists=True), help="Path to SoundFont (.sf2) file"
)
def cli(sf2):
    root = tk.Tk()
    app = MidiTaggerApp(root, soundfont_path=sf2)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    cli()
